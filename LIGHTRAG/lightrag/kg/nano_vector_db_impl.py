import asyncio
import os
from typing import Any, final
from dataclasses import dataclass
import numpy as np

import time

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
import pipmaster as pm
from lightrag.base import (
    BaseVectorStorage,
)

if not pm.is_installed("nano-vectordb"):
    pm.install("nano-vectordb")

try:
    from nano_vectordb import NanoVectorDB
except ImportError as e:
    raise ImportError(
        "`nano-vectordb` library is not installed. Please install it via pip: `pip install nano-vectordb`."
    ) from e


@final
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    def __post_init__(self):
        # Initialize lock only for file operations
        self._save_lock = asyncio.Lock()
        # Use global config value if specified, otherwise use default
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        # print("dm m có gọi t ra mà")
        current_time = time.time()
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        
        # print(len(embeddings), len(list_data))
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
                # print(d["__vector__"])
                # print(embeddings[i])
                # print(d)
            results = self._client.upsert(datas=list_data)
            async with self._save_lock:
                self._client.save()
            logger.info(f"Successfully saved {len(list_data)} vectors to {self.namespace}")
            return results        
            # return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {
                **dp,
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            self._client.delete(ids)
            logger.info(
                f"Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )
            # Check if the entity exists
            if self._client.get([entity_id]):
                await self.delete([entity_id])
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found in storage")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            logger.debug(f"Found {len(relations)} relations for entity {entity_name}")
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                await self.delete(ids_to_delete)
                logger.debug(
                    f"Deleted {len(ids_to_delete)} relations for {entity_name}"
                )
            else:
                logger.debug(f"No relations found for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def index_done_callback(self) -> None:
        async with self._save_lock:
            self._client.save()

    async def get_entity_embedding(self, entity_name: str) -> np.ndarray | None:
        """Get embedding vector for an entity by name
        
        Args:
            entity_name (str): Tên entity cần lấy embedding
            
        Returns:
            np.ndarray | None: Vector embedding hoặc None nếu không tìm thấy
        """
        # Chuẩn hóa entity name
        entity_name = f'"{entity_name.upper()}"' if not entity_name.startswith('"') else entity_name.upper()
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        
        try:
            # Lấy data object từ nano-vectordb
            data = self._client.get([entity_id])
            
            if data and len(data) > 0:
                # Tìm chỉ mục của entity trong storage data
                index = None
                for i, item in enumerate(self._client._NanoVectorDB__storage["data"]):
                    if item["__id__"] == entity_id:
                        index = i
                        break
                
                if index is not None:
                    # Lấy vector từ matrix theo index
                    vector = self._client._NanoVectorDB__storage["matrix"][index]
                    logger.debug(f"Found embedding for entity: {entity_name}")
                    return vector
                else:
                    logger.warning(f"Entity found but no embedding vector: {entity_name}")
            else:
                logger.warning(f"Entity not found: {entity_name}")
        except Exception as e:
            logger.error(f"Error getting embedding for entity {entity_name}: {e}")
        
        return None

    async def get_entity_embedding_by_id(self, entity_id: str) -> np.ndarray | None:
        """Get embedding vector for an entity by name
        
        Args:
            entity_name (str): Tên entity cần lấy embedding
            
        Returns:
            np.ndarray | None: Vector embedding hoặc None nếu không tìm thấy
        """        
        try:
            # Lấy data object từ nano-vectordb
            data = self._client.get([entity_id])
            
            if data and len(data) > 0:
                # Tìm chỉ mục của entity trong storage data
                index = None
                for i, item in enumerate(self._client._NanoVectorDB__storage["data"]):
                    if item["__id__"] == entity_id:
                        index = i
                        break
                
                if index is not None:
                    # Lấy vector từ matrix theo index
                    vector = self._client._NanoVectorDB__storage["matrix"][index]
                    logger.debug(f"Found embedding for entity: {entity_id}")
                    return vector
                else:
                    logger.warning(f"Entity found but no embedding vector: {entity_id}")
            else:
                logger.warning(f"Entity not found: {entity_id}")
        except Exception as e:
            logger.error(f"Error getting embedding for entity {entity_id}: {e}")
        
        return None

    async def get_relation_embedding(self, src_id: str, tgt_id: str) -> np.ndarray | None:
        """Get embedding vector for a relation between two entities
        
        Args:
            src_id (str): Tên entity nguồn
            tgt_id (str): Tên entity đích
            
        Returns:
            np.ndarray | None: Vector embedding hoặc None nếu không tìm thấy
        """
        # Chuẩn hóa entity names
        src_id = f'"{src_id.upper()}"' if not src_id.startswith('"') else src_id.upper()
        tgt_id = f'"{tgt_id.upper()}"' if not tgt_id.startswith('"') else tgt_id.upper()
        
        # Tạo relation ID
        relation_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        
        try:
            # Tìm kiếm relation theo ID
            index = None
            for i, item in enumerate(self._client._NanoVectorDB__storage["data"]):
                if item["__id__"] == relation_id:
                    index = i
                    break
            
            if index is not None:
                # Lấy vector từ matrix theo index
                vector = self._client._NanoVectorDB__storage["matrix"][index]
                return vector
            else:
                # Thử tìm theo chiều ngược lại
                reverse_relation_id = compute_mdhash_id(tgt_id + src_id, prefix="rel-")
                
                for i, item in enumerate(self._client._NanoVectorDB__storage["data"]):
                    if item["__id__"] == reverse_relation_id:
                        vector = self._client._NanoVectorDB__storage["matrix"][i]
                        return vector
                        
                logger.warning(f"Relation between {src_id} and {tgt_id} not found")
        except Exception as e:
            logger.error(f"Error getting embedding for relation between {src_id} and {tgt_id}: {e}")
        
        return None

    async def get_chunk_embedding(self, chunk_id: str) -> np.ndarray | None:
        """Get embedding vector for a text chunk
        
        Args:
            chunk_id (str): ID của chunk cần lấy embedding
            
        Returns:
            np.ndarray | None: Vector embedding hoặc None nếu không tìm thấy
        """
        try:
            # Lấy data object từ nano-vectordb
            data = self._client.get([chunk_id])
            
            if data and len(data) > 0:
                # Tìm chỉ mục của chunk trong storage data
                index = None
                for i, item in enumerate(self._client._NanoVectorDB__storage["data"]):
                    if item["__id__"] == chunk_id:
                        index = i
                        break
                
                if index is not None:
                    # Lấy vector từ matrix theo index
                    vector = self._client._NanoVectorDB__storage["matrix"][index]
                    logger.debug(f"Found embedding for chunk: {chunk_id}")
                    return vector
                else:
                    logger.warning(f"Chunk found but no embedding vector: {chunk_id}")
            else:
                logger.warning(f"Chunk not found: {chunk_id}")
        except Exception as e:
            logger.error(f"Error getting embedding for chunk {chunk_id}: {e}")
        
        return None

    async def batch_get_embeddings(self, ids: list[str]) -> dict[str, np.ndarray]:
        """Get embeddings for multiple items by their IDs
        
        Args:
            ids (list[str]): Danh sách các ID cần lấy embedding
            
        Returns:
            dict[str, np.ndarray]: Dictionary {id: embedding_vector}
        """
        results = {}
        
        try:
            data_objects = self._client.get(ids)
            id_to_index = {}
            
            # Tạo mapping từ ID đến index trong matrix
            for i, item in enumerate(self._client._NanoVectorDB__storage["data"]):
                if item["__id__"] in ids:
                    id_to_index[item["__id__"]] = i
            
            # Lấy embedding từ matrix
            for id_value in ids:
                if id_value in id_to_index:
                    index = id_to_index[id_value]
                    results[id_value] = self._client._NanoVectorDB__storage["matrix"][index]
                else:
                    logger.warning(f"No embedding found for ID: {id_value}")
        except Exception as e:
            logger.error(f"Error in batch getting embeddings: {e}")
        
        return results

    async def force_save(self) -> None:
        """Force save the vector database to disk"""
        async with self._save_lock:
            self._client.save()
            logger.info(f"Vector storage {self.namespace} saved to {self._client_file_name}")
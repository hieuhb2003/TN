from __future__ import annotations

import asyncio
import os
import configparser
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterator, cast
from sklearn.metrics.pairwise import cosine_similarity  
import numpy as np        
import re
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
)
from .namespace import NameSpace, make_namespace
from .operate import (
    chunking_by_token_size,
    extract_entities,
    extract_keywords_only,
    kg_query,
    kg_query_with_keywords,
    mix_kg_vector_query,
    naive_query,
    kg_retrieval
)
from .prompt import GRAPH_FIELD_SEP
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    convert_response_to_json,
    limit_async_func_call,
    logger,
    set_logger,
    detect_language,
)
from .types import KnowledgeGraph

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Storage type and implementation compatibility validation table
STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": [
            "JsonKVStorage",
            "MongoKVStorage",
            "RedisKVStorage",
            "TiDBKVStorage",
            "PGKVStorage",
            "OracleKVStorage",
        ],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": [
            "NetworkXStorage",
            "Neo4JStorage",
            "MongoGraphStorage",
            "TiDBGraphStorage",
            "AGEStorage",
            "GremlinStorage",
            "PGGraphStorage",
            "OracleGraphStorage",
        ],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    "VECTOR_STORAGE": {
        "implementations": [
            "NanoVectorDBStorage",
            "MilvusVectorDBStorage",
            "ChromaVectorDBStorage",
            "TiDBVectorDBStorage",
            "PGVectorStorage",
            "FaissVectorDBStorage",
            "QdrantVectorDBStorage",
            "OracleVectorDBStorage",
            "MongoVectorDBStorage",
        ],
        "required_methods": ["query", "upsert"],
    },
    "DOC_STATUS_STORAGE": {
        "implementations": [
            "JsonDocStatusStorage",
            "PGDocStatusStorage",
            "PGDocStatusStorage",
            "MongoDocStatusStorage",
        ],
        "required_methods": ["get_docs_by_status"],
    },
}

# Storage implementation environment variable without default value
STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    # KV Storage Implementations
    "JsonKVStorage": [],
    "MongoKVStorage": [],
    "RedisKVStorage": ["REDIS_URI"],
    "TiDBKVStorage": ["TIDB_USER", "TIDB_PASSWORD", "TIDB_DATABASE"],
    "PGKVStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "OracleKVStorage": [
        "ORACLE_DSN",
        "ORACLE_USER",
        "ORACLE_PASSWORD",
        "ORACLE_CONFIG_DIR",
    ],
    # Graph Storage Implementations
    "NetworkXStorage": [],
    "Neo4JStorage": ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"],
    "MongoGraphStorage": [],
    "TiDBGraphStorage": ["TIDB_USER", "TIDB_PASSWORD", "TIDB_DATABASE"],
    "AGEStorage": [
        "AGE_POSTGRES_DB",
        "AGE_POSTGRES_USER",
        "AGE_POSTGRES_PASSWORD",
    ],
    "GremlinStorage": ["GREMLIN_HOST", "GREMLIN_PORT", "GREMLIN_GRAPH"],
    "PGGraphStorage": [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    ],
    "OracleGraphStorage": [
        "ORACLE_DSN",
        "ORACLE_USER",
        "ORACLE_PASSWORD",
        "ORACLE_CONFIG_DIR",
    ],
    # Vector Storage Implementations
    "NanoVectorDBStorage": [],
    "MilvusVectorDBStorage": [],
    "ChromaVectorDBStorage": [],
    "TiDBVectorDBStorage": ["TIDB_USER", "TIDB_PASSWORD", "TIDB_DATABASE"],
    "PGVectorStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "FaissVectorDBStorage": [],
    "QdrantVectorDBStorage": ["QDRANT_URL"],  # QDRANT_API_KEY has default value None
    "OracleVectorDBStorage": [
        "ORACLE_DSN",
        "ORACLE_USER",
        "ORACLE_PASSWORD",
        "ORACLE_CONFIG_DIR",
    ],
    "MongoVectorDBStorage": [],
    # Document Status Storage Implementations
    "JsonDocStatusStorage": [],
    "PGDocStatusStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "MongoDocStatusStorage": [],
}

# Storage implementation module mapping
STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "JsonDocStatusStorage": ".kg.json_doc_status_impl",
    "Neo4JStorage": ".kg.neo4j_impl",
    "OracleKVStorage": ".kg.oracle_impl",
    "OracleGraphStorage": ".kg.oracle_impl",
    "OracleVectorDBStorage": ".kg.oracle_impl",
    "MilvusVectorDBStorage": ".kg.milvus_impl",
    "MongoKVStorage": ".kg.mongo_impl",
    "MongoDocStatusStorage": ".kg.mongo_impl",
    "MongoGraphStorage": ".kg.mongo_impl",
    "MongoVectorDBStorage": ".kg.mongo_impl",
    "RedisKVStorage": ".kg.redis_impl",
    "ChromaVectorDBStorage": ".kg.chroma_impl",
    "TiDBKVStorage": ".kg.tidb_impl",
    "TiDBVectorDBStorage": ".kg.tidb_impl",
    "TiDBGraphStorage": ".kg.tidb_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "AGEStorage": ".kg.age_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "GremlinStorage": ".kg.gremlin_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
    "FaissVectorDBStorage": ".kg.faiss_impl",
    "QdrantVectorDBStorage": ".kg.qdrant_impl",
}


def lazy_external_import(module_name: str, class_name: str) -> Callable[..., Any]:
    """Lazily import a class from an external module based on the package of the caller."""
    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args: Any, **kwargs: Any):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class LightRAG:
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    """Directory where cache and temporary files are stored."""

    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache.
    - enabled: If True, enables caching to avoid redundant computations.
    - similarity_threshold: Minimum similarity score to use cached embeddings.
    - use_llm_check: If True, validates cached embeddings using an LLM.
    """

    kv_storage: str = field(default="JsonKVStorage")
    """Storage backend for key-value data."""

    vector_storage: str = field(default="NanoVectorDBStorage")
    """Storage backend for vector embeddings."""

    graph_storage: str = field(default="NetworkXStorage")
    """Storage backend for knowledge graphs."""

    doc_status_storage: str = field(default="JsonDocStatusStorage")
    """Storage type for tracking document processing statuses."""

    # Logging
    current_log_level = logger.level
    log_level: int = field(default=current_log_level)
    """Logging level for the system (e.g., 'DEBUG', 'INFO', 'WARNING')."""

    log_dir: str = field(default=os.getcwd())
    """Directory where logs are stored. Defaults to the current working directory."""

    # Text chunking
    chunk_token_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    """Maximum number of tokens per text chunk when splitting documents."""

    chunk_overlap_token_size: int = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))
    """Number of overlapping tokens between consecutive text chunks to preserve context."""

    tiktoken_model_name: str = "gpt-4o-mini"
    """Model name used for tokenization when chunking text."""

    # Entity extraction
    entity_extract_max_gleaning: int = 1
    """Maximum number of entity extraction attempts for ambiguous content."""

    entity_summary_to_max_tokens: int = int(os.getenv("MAX_TOKEN_SUMMARY", "500"))
    """Maximum number of tokens used for summarizing extracted entities."""

    # Node embedding
    node_embedding_algorithm: str = "node2vec"
    """Algorithm used for node embedding in knowledge graphs."""

    node2vec_params: dict[str, int] = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    """Configuration for the node2vec embedding algorithm:
    - dimensions: Number of dimensions for embeddings.
    - num_walks: Number of random walks per node.
    - walk_length: Number of steps per random walk.
    - window_size: Context window size for training.
    - iterations: Number of iterations for training.
    - random_seed: Seed value for reproducibility.
    """

    embedding_func: EmbeddingFunc | None = None
    """Function for computing text embeddings. Must be set before use."""

    embedding_batch_num: int = 32
    """Batch size for embedding computations."""

    embedding_func_max_async: int = 16
    """Maximum number of concurrent embedding function calls."""

    # LLM Configuration
    llm_model_func: Callable[..., object] | None = None
    """Function for interacting with the large language model (LLM). Must be set before use."""

    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    """Name of the LLM model used for generating responses."""

    llm_model_max_token_size: int = int(os.getenv("MAX_TOKENS", "32768"))
    """Maximum number of tokens allowed per LLM response."""

    llm_model_max_async: int = int(os.getenv("MAX_ASYNC", "16"))
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    # Storage
    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    namespace_prefix: str = field(default="")
    """Prefix for namespacing stored data across different environments."""

    enable_llm_cache: bool = True
    """Enables caching for LLM responses to avoid redundant computations."""

    enable_llm_cache_for_entity_extract: bool = True
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    # Extensions
    addon_params: dict[str, Any] = field(default_factory=dict)

    # Storages Management
    auto_manage_storages_states: bool = True
    """If True, lightrag will automatically calls initialize_storages and finalize_storages at the appropriate times."""

    """Dictionary for additional parameters and extensions."""
    convert_response_to_json_func: Callable[[str], dict[str, Any]] = (
        convert_response_to_json
    )

    # Custom Chunking Function
    chunking_func: Callable[
        [
            str,
            str | None,
            bool,
            int,
            int,
            str,
        ],
        list[dict[str, Any]],
    ] = chunking_by_token_size

    def verify_storage_implementation(
        self, storage_type: str, storage_name: str
    ) -> None:
        """Verify if storage implementation is compatible with specified storage type

        Args:
            storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
            storage_name: Storage implementation name

        Raises:
            ValueError: If storage implementation is incompatible or missing required methods
        """
        if storage_type not in STORAGE_IMPLEMENTATIONS:
            raise ValueError(f"Unknown storage type: {storage_type}")

        storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
        if storage_name not in storage_info["implementations"]:
            raise ValueError(
                f"Storage implementation '{storage_name}' is not compatible with {storage_type}. "
                f"Compatible implementations are: {', '.join(storage_info['implementations'])}"
            )

    def check_storage_env_vars(self, storage_name: str) -> None:
        """Check if all required environment variables for storage implementation exist

        Args:
            storage_name: Storage implementation name

        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            raise ValueError(
                f"Storage implementation '{storage_name}' requires the following "
                f"environment variables: {', '.join(missing_vars)}"
            )

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, "lightrag.log")
        set_logger(log_file)

        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            self.verify_storage_implementation(storage_type, storage_name)
            # Check environment variables
            # self.check_storage_env_vars(storage_name)

        # Ensure vector_db_storage_cls_kwargs has required fields
        default_vector_db_kwargs = {
            "cosine_better_than_threshold": float(os.getenv("COSINE_THRESHOLD", "0.2"))
        }
        self.vector_db_storage_cls_kwargs = {
            **default_vector_db_kwargs,
            **self.vector_db_storage_cls_kwargs,
        }

        # Life cycle
        self.storages_status = StoragesStatus.NOT_CREATED

        # Show config
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Init LLM
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(  # type: ignore
            self.embedding_func
        )

        # Initialize all storages
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )  # type: ignore
        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls, global_config=global_config
        )
        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls, global_config=global_config
        )

        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
        )
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
        )

        if self.llm_response_cache and hasattr(
            self.llm_response_cache, "global_config"
        ):
            hashing_kv = self.llm_response_cache
        else:
            hashing_kv = self.key_string_value_json_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,  # type: ignore
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )

        self.storages_status = StoragesStatus.CREATED

        # Initialize storages
        if self.auto_manage_storages_states:
            loop = always_get_an_event_loop()
            loop.run_until_complete(self.initialize_storages())

    def __del__(self):
        # Finalize storages
        # if self.auto_manage_storages_states:
        #     print("Debug info:")
        #     print(f"llm_model_func is None: {self.llm_model_func is None}")
        #     print(f"embedding_func is None: {self.embedding_func is None}")
        #     loop = always_get_an_event_loop()
        #     loop.run_until_complete(self.finalize_storages())
        pass

    async def initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self.storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self.storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    async def finalize_storages(self):
        """Asynchronously finalize the storages"""
        if self.storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.finalize())

            await asyncio.gather(*tasks)

            self.storages_status = StoragesStatus.FINALIZED
            logger.debug("Finalized Storages")

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self, nodel_label: str, max_depth: int
    ) -> KnowledgeGraph:
        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label=nodel_label, max_depth=max_depth
        )

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        language: str = "Vietnamese" ,
        matching_method: str = "hybrid" 
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """
        self.addon_params["current_language"] = language
        
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(input, split_by_character, split_by_character_only, language, matching_method)
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        language: str = "Vietnamese" , # Add language parameter
        matching_method: str = "hybrid"
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """
        self.addon_params["current_language"] = language
        self.addon_params["matching_method"] = matching_method
        
        await self.apipeline_enqueue_documents(input)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

    def insert_custom_chunks(self, full_text: str, text_chunks: list[str]) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_custom_chunks(full_text, text_chunks))

    async def ainsert_custom_chunks(
        self, full_text: str, text_chunks: list[str]
    ) -> None:
        update_storage = False
        try:
            doc_key = compute_mdhash_id(full_text.strip(), prefix="doc-")
            new_docs = {doc_key: {"content": full_text.strip()}}

            _add_doc_keys = await self.full_docs.filter_keys(set(doc_key))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("This document is already in the storage.")
                return

            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks: dict[str, Any] = {}
            for chunk_text in text_chunks:
                chunk_text_stripped = chunk_text.strip()
                chunk_key = compute_mdhash_id(chunk_text_stripped, prefix="chunk-")

                inserting_chunks[chunk_key] = {
                    "content": chunk_text_stripped,
                    "full_doc_id": doc_key,
                }

            doc_ids = set(inserting_chunks.keys())
            add_chunk_keys = await self.text_chunks.filter_keys(doc_ids)
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage.")
                return

            tasks = [
                self.chunks_vdb.upsert(inserting_chunks),
                self._process_entity_relation_graph(inserting_chunks),
                self.full_docs.upsert(new_docs),
                self.text_chunks.upsert(inserting_chunks),
            ]
            await asyncio.gather(*tasks)

        finally:
            if update_storage:
                await self._insert_done()

    async def apipeline_enqueue_documents(self, input: str | list[str]) -> None:
        """
        Pipeline for Processing Documents

        1. Remove duplicate contents from the list
        2. Generate document IDs and initial status
        3. Filter out already processed documents
        4. Enqueue document in status
        """
        if isinstance(input, str):
            input = [input]

        # 1. Remove duplicate contents from the list
        unique_contents = list(set(doc.strip() for doc in input))

        # 2. Generate document IDs and initial status
        new_docs: dict[str, Any] = {
            compute_mdhash_id(content, prefix="doc-"): {
                "content": content,
                "content_summary": self._get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for content in unique_contents
        }

        # 3. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)
        # Filter new_docs to only include documents with unique IDs
        new_docs = {doc_id: new_docs[doc_id] for doc_id in unique_new_doc_ids}

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        # 4. Store status document
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Split document content into chunks
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """
        # 1. Get all pending, failed, and abnormally terminated processing documents.
        to_process_docs: dict[str, DocProcessingStatus] = {}

        processing_docs = await self.doc_status.get_docs_by_status(DocStatus.PROCESSING)
        to_process_docs.update(processing_docs)
        failed_docs = await self.doc_status.get_docs_by_status(DocStatus.FAILED)
        to_process_docs.update(failed_docs)
        pendings_docs = await self.doc_status.get_docs_by_status(DocStatus.PENDING)
        to_process_docs.update(pendings_docs)

        if not to_process_docs:
            logger.info("All documents have been processed or are duplicates")
            return

        # 2. split docs into chunks, insert chunks, update doc status
        batch_size = self.addon_params.get("insert_batch_size", 10)
        docs_batches = [
            list(to_process_docs.items())[i : i + batch_size]
            for i in range(0, len(to_process_docs), batch_size)
        ]

        logger.info(f"Number of batches to process: {len(docs_batches)}.")

        # 3. iterate over batches
        for batch_idx, docs_batch in enumerate(docs_batches):
            # 4. iterate over batch
            for doc_id_processing_status in docs_batch:
                doc_id, status_doc = doc_id_processing_status
                # Update status in processing
                doc_status_id = compute_mdhash_id(status_doc.content, prefix="doc-")
                await self.doc_status.upsert(
                    {
                        doc_status_id: {
                            "status": DocStatus.PROCESSING,
                            "updated_at": datetime.now().isoformat(),
                            "content": status_doc.content,
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                        }
                    }
                )
                # Generate chunks from document
                chunks: dict[str, Any] = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_id,
                    }
                    for dp in self.chunking_func(
                        status_doc.content,
                        split_by_character,
                        split_by_character_only,
                        self.chunk_overlap_token_size,
                        self.chunk_token_size,
                        self.tiktoken_model_name,
                    )
                }

                # Process document (text chunks and full docs) in parallel
                tasks = [
                    self.chunks_vdb.upsert(chunks),
                    self._process_entity_relation_graph(chunks),
                    self.full_docs.upsert({doc_id: {"content": status_doc.content}}),
                    self.text_chunks.upsert(chunks),
                ]
                try:
                    await asyncio.gather(*tasks)
                    await self.doc_status.upsert(
                        {
                            doc_status_id: {
                                "status": DocStatus.PROCESSED,
                                "chunks_count": len(chunks),
                                "content": status_doc.content,
                                "content_summary": status_doc.content_summary,
                                "content_length": status_doc.content_length,
                                "created_at": status_doc.created_at,
                                "updated_at": datetime.now().isoformat(),
                            }
                        }
                    )
                    await self._insert_done()

                except Exception as e:
                    logger.error(f"Failed to process document {doc_id}: {str(e)}")
                    await self.doc_status.upsert(
                        {
                            doc_status_id: {
                                "status": DocStatus.FAILED,
                                "error": str(e),
                                "content": status_doc.content,
                                "content_summary": status_doc.content_summary,
                                "content_length": status_doc.content_length,
                                "created_at": status_doc.created_at,
                                "updated_at": datetime.now().isoformat(),
                            }
                        }
                    )
                    continue
            logger.info(f"Completed batch {batch_idx + 1} of {len(docs_batches)}.")

    async def _process_entity_relation_graph(self, chunk: dict[str, Any]) -> None:
        try:
            new_kg = await extract_entities(
                chunk,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                llm_response_cache=self.llm_response_cache,
                global_config=asdict(self),
            )
            if new_kg is None:
                logger.info("No new entities or relationships extracted.")
            else:
                logger.info("New entities or relationships extracted.")
                self.chunk_entity_relation_graph = new_kg
                # Get newly extracted entities from this chunk

                extracted_entities = []
                for node in self.chunk_entity_relation_graph._graph.nodes():
                    node_data = self.chunk_entity_relation_graph._graph.nodes[node]
                    if "source_id" in node_data:
                        source_ids = node_data["source_id"].split("|") if "|" in node_data["source_id"] else [node_data["source_id"]]
                        # Check if any of the chunk keys are in source_ids
                        if any(chunk_id in chunk for chunk_id in source_ids):
                            extracted_entities.append(node)
                
                if extracted_entities:
                    # Get default language from addon_params
                    current_language = self.addon_params.get("current_language", "Vietnamese")
                    # Try to link with existing entities in other languages
                    await self._link_cross_lingual_entities(extracted_entities, current_language, matching_method=self.addon_params["matching_method"])

        except Exception as e:
            logger.error(f"Failed to extract entities and relationships: {e}")
            raise e
    
    # async def _link_cross_lingual_entities(
    #     self, 
    #     new_entities: list[str], 
    #     source_language: str = "Vietnamese",
    #     matching_method: str = "hybrid"  # Thêm tham số này
    # ) -> None:
    #     """
    #     Try to find matches for new entities in other languages and create cross-lingual links
    #     """
    #     if not new_entities:
    #         return
            
    #     logger.info(f"Looking for cross-lingual matches for {len(new_entities)} new entities")
        
    #     # Tìm các entities trong các ngôn ngữ khác
    #     existing_entities = []
    #     for node in self.chunk_entity_relation_graph._graph.nodes():
    #         if node in new_entities:
    #             continue
                
    #         node_data = self.chunk_entity_relation_graph._graph.nodes[node]
    #         entity_language = node_data.get("language", "")
            
    #         # Xác định ngôn ngữ của entity
    #         if not entity_language:
    #             for u, v, edge_data in self.chunk_entity_relation_graph._graph.edges(data=True):
    #                 if u == node or v == node:
    #                     if edge_data.get("relation_type") == "translation_equivalent":
    #                         if edge_data.get("original_language") and edge_data.get("original_language") != source_language:
    #                             entity_language = edge_data.get("original_language")
    #                             break
    #                         elif edge_data.get("translated_language") and edge_data.get("translated_language") != source_language:
    #                             entity_language = edge_data.get("translated_language")
    #                             break
            
    #         if entity_language and entity_language != source_language:
    #             existing_entities.append((node, entity_language))
        
    #     # Group entities by language
    #     entities_by_language = {}
    #     for entity, language in existing_entities:
    #         if language not in entities_by_language:
    #             entities_by_language[language] = []
    #         entities_by_language[language].append(entity)
        
    #     # Match và link cho từng ngôn ngữ
    #     for target_language, target_entities in entities_by_language.items():
    #         entity_pairs = await self._match_entities_for_linking(
    #             new_entities, 
    #             target_entities,
    #             source_language,
    #             target_language,
    #             matching_method=matching_method  # Truyền method vào
    #         )
            
    #         if entity_pairs:
    #             await self._add_cross_lingual_links(
    #                 entity_pairs, 
    #                 source_language, 
    #                 target_language
    #             )
    async def _link_cross_lingual_entities(
        self, 
        new_entities: list[str], 
        source_language: str = "Vietnamese",
        matching_method: str = "hybrid"
    ) -> None:
        if not new_entities:
            return
                
        logger.info(f"Looking for cross-lingual matches for {len(new_entities)} new entities")
        
        # Track đã match để tránh match lại
        already_matched_entities = set()
        
        # Lấy tất cả các translation links hiện có
        for u, v, edge_data in self.chunk_entity_relation_graph._graph.edges(data=True):
            if edge_data.get("relation_type") == "translation_equivalent":
                already_matched_entities.add(u)
                already_matched_entities.add(v)
        print()
        # Lọc ra các entities chưa có translation
        new_entities = [e for e in new_entities if e not in already_matched_entities]
        
        if not new_entities:
            logger.info("All entities already have translations")
            return
        
        # Tìm các entities trong các ngôn ngữ khác
        existing_entities = []
        existing_entity_languages = {}
        
        for node in self.chunk_entity_relation_graph._graph.nodes():
            if node in new_entities:
                continue
                
            node_data = self.chunk_entity_relation_graph._graph.nodes[node]
            entity_language = node_data.get("language", "")
            
            # Xác định ngôn ngữ của entity
            if not entity_language:
                entity_language = self._detect_entity_language(node)
            
            if entity_language and entity_language != source_language:
                existing_entities.append(node)
                existing_entity_languages[node] = entity_language
        
        # Group entities by language
        entities_by_language = {}
        for entity in existing_entities:
            language = existing_entity_languages[entity]
            if language not in entities_by_language:
                entities_by_language[language] = []
            entities_by_language[language].append(entity)
        
        # Match và link cho từng ngôn ngữ
        for target_language, target_entities in entities_by_language.items():
            entity_pairs = await self._match_entities_for_linking(
                new_entities, 
                target_entities,
                source_language,
                target_language,
                matching_method=matching_method
            )
            
            if entity_pairs:
                await self._add_cross_lingual_links(
                    entity_pairs, 
                    source_language, 
                    target_language
                )

    async def _detect_entity_language(self, entity: str) -> str:
        """Helper function to detect entity language from its edges"""
        for u, v, edge_data in self.chunk_entity_relation_graph._graph.edges(data=True):
            if u == entity or v == entity:
                if edge_data.get("relation_type") == "translation_equivalent":
                    if edge_data.get("original_language"):
                        return edge_data.get("original_language")
                    elif edge_data.get("translated_language"):
                        return edge_data.get("translated_language")
        return ""
    async def _match_entities_for_linking(
        self, 
        source_entities: list[str], 
        target_entities: list[str],
        source_language: str,
        target_language: str,
        similarity_threshold: float = 0.80,
        matching_method: str = "hybrid"  # Có thể là "embedding", "llm", hoặc "hybrid"
    ) -> list[tuple[str, str]]:
        """
        Match entities between languages using specified method(s)
        
        Args:
            matching_method: 
                - "embedding": Chỉ dùng embedding similarity
                - "llm": Chỉ dùng LLM
                - "hybrid": Kết hợp cả embedding và LLM
        """
        matched_pairs = []
        matched_sources = set()

        # Step 1: Exact matching (luôn thực hiện trước)
        for source_entity in source_entities:
            source_name = source_entity.strip('"').upper()
            for target_entity in target_entities:
                target_name = target_entity.strip('"').upper()
                if source_name == target_name:
                    matched_pairs.append((source_entity, target_entity))
                    matched_sources.add(source_entity)
                    break

        # Lấy các entities chưa được match
        unmatched_sources = [e for e in source_entities if e not in matched_sources]
        
        if not unmatched_sources:
            return matched_pairs

        # Step 2: Embedding-based matching
        if matching_method in ["embedding", "hybrid"] and self.embedding_func:
            embedding_matches = await self._match_entities_embedding(
                unmatched_sources,
                target_entities,
                source_language,
                target_language,
                similarity_threshold
            )
            
            for source, target in embedding_matches:
                matched_pairs.append((source, target))
                matched_sources.add(source)
                logger.info(f"Embedding matched: {source} with {target}")

        # Step 3: LLM-based matching
        if matching_method in ["llm", "hybrid"] and self.llm_model_func:
            # Chỉ xử lý các entities chưa được match bởi embedding
            still_unmatched = [e for e in unmatched_sources if e not in matched_sources]
            
            llm_matches = await self._match_entities_llm(
                still_unmatched,
                target_entities,
                source_language,
                target_language
            )
            
            matched_pairs.extend(llm_matches)

        return matched_pairs

    async def _match_entities_embedding(
        self,
        source_entities: list[str],
        target_entities: list[str],
        source_language: str,
        target_language: str,
        similarity_threshold: float = 0.80
    ) -> list[tuple[str, str]]:
        """Match entities using embedding similarity directly from vector database"""
        matches = []
        match_similarities = []  # Store similarity scores
        
        # Format entity names correctly for proper ID generation
        source_entities_formatted = [f'"{e.strip().upper()}"' if not e.strip().startswith('"') else e.strip() for e in source_entities]
        target_entities_formatted = [f'"{e.strip().upper()}"' if not e.strip().startswith('"') else e.strip() for e in target_entities]
        
        # Get entity IDs
        source_entity_ids = [compute_mdhash_id(e, prefix="ent-") for e in source_entities_formatted]
        target_entity_ids = [compute_mdhash_id(e, prefix="ent-") for e in target_entities_formatted]
        
        # Fetch embeddings directly from vector database
        source_embeddings = []
        valid_source_indices = []
        
        for i, entity_id in enumerate(source_entity_ids):
            vector = await self.entities_vdb.get_entity_embedding_by_id(entity_id)
            if vector is not None:
                source_embeddings.append(vector)
                valid_source_indices.append(i)
        
        target_embeddings = []
        valid_target_indices = []
        
        for i, entity_id in enumerate(target_entity_ids):
            vector = await self.entities_vdb.get_entity_embedding_by_id(entity_id)
            if vector is not None:
                target_embeddings.append(vector)
                valid_target_indices.append(i)
        
        if not source_embeddings or not target_embeddings:
            logger.warning("No valid entity embeddings found for comparison")
            return matches

        # Convert to numpy arrays for similarity computation
        source_embeddings_np = np.array(source_embeddings)
        target_embeddings_np = np.array(target_embeddings)
        
        # Compute cosine similarity
        similarities = cosine_similarity(source_embeddings_np, target_embeddings_np)
        
        # Find matches based on similarity threshold
        for i, source_idx in enumerate(valid_source_indices):
            max_sim_idx = similarities[i].argmax()
            if similarities[i][max_sim_idx] >= similarity_threshold:
                source_entity = source_entities[source_idx]
                target_entity = target_entities[valid_target_indices[max_sim_idx]]
                similarity_score = float(similarities[i][max_sim_idx])
                
                # Add to results for high similarity
                matches.append((source_entity, target_entity))
                match_similarities.append(similarity_score)  # Store the similarity score
                
                logger.info(
                    f"Matched entities: {source_entity} <-> {target_entity}\n"
                    f"Similarity: {similarity_score:.3f}"
                )
                
                # Get node data for logging
                source_node_data = await self.chunk_entity_relation_graph.get_node(source_entities_formatted[source_idx])
                target_node_data = await self.chunk_entity_relation_graph.get_node(target_entities_formatted[valid_target_indices[max_sim_idx]])
                
                if source_node_data and target_node_data:
                    source_desc = source_node_data.get("description", "No description")
                    target_desc = target_node_data.get("description", "No description")
                    logger.debug(
                        f"Language pair: {source_language} -> {target_language}\n"
                        f"Source description: {source_desc[:100]}...\n"
                        f"Target description: {target_desc[:100]}..."
                    )
        
        # Store matched pairs in JSON if matches were found
        if matches:
            await self._store_matched_pairs_json(
                matches,
                match_similarities,  # Pass the similarity scores
                source_entities_formatted,
                target_entities_formatted,
                valid_source_indices,
                valid_target_indices,
                source_language,
                target_language
            )
            
        return matches

    async def _store_matched_pairs_json(
        self,
        matches: list[tuple[str, str]],
        match_similarities: list[float],  # Add similarity scores parameter
        source_entities_formatted: list[str],
        target_entities_formatted: list[str],
        valid_source_indices: list[int],
        valid_target_indices: list[int],
        source_language: str,
        target_language: str
    ) -> None:
        """
        Store matched entity pairs in a JSON file
        
        Args:
            matches: List of matched entity pairs (source_entity, target_entity)
            match_similarities: List of similarity scores for each matched pair
            source_entities_formatted: Formatted source entity names
            target_entities_formatted: Formatted target entity names
            valid_source_indices: Valid indices for source entities
            valid_target_indices: Valid indices for target entities
            source_language: Source language name
            target_language: Target language name
        """
        import os
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.working_dir, "matched_pairs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"matched_pairs_{source_language}_{target_language}_{timestamp}.json")
        
        # Create dictionary to map source entities to their indices
        source_entity_idx_map = {}
        for idx, entity_idx in enumerate(valid_source_indices):
            source_entity = source_entities_formatted[entity_idx].strip('"')
            source_entity_idx_map[source_entity] = idx
            
        # Create dictionary to map target entities to their indices
        target_entity_idx_map = {}
        for idx, entity_idx in enumerate(valid_target_indices):
            target_entity = target_entities_formatted[entity_idx].strip('"')
            target_entity_idx_map[target_entity] = idx
        
        # Create list to store pairs with descriptions
        pairs_with_descriptions = []
        
        for i, (source_entity, target_entity) in enumerate(matches):
            # Get formatted entity names
            source_entity_formatted = source_entities_formatted[
                valid_source_indices[source_entity_idx_map.get(source_entity.strip('"'), 0)]
            ]
            target_entity_formatted = target_entities_formatted[
                valid_target_indices[target_entity_idx_map.get(target_entity.strip('"'), 0)]
            ]
            
            # Get entity data from graph
            source_node_data = await self.chunk_entity_relation_graph.get_node(source_entity_formatted)
            target_node_data = await self.chunk_entity_relation_graph.get_node(target_entity_formatted)
            
            # Prepare pair data with descriptions
            pair_data = {
                "original_entity": {
                    "name": source_entity.strip('"'),
                    "description": source_node_data.get("description", "No description") if source_node_data else "No description",
                    "language": source_language
                },
                "translated_entity": {
                    "name": target_entity.strip('"'),
                    "description": target_node_data.get("description", "No description") if target_node_data else "No description",
                    "language": target_language
                },
                "similarity_score": match_similarities[i]  # Use the actual similarity score
            }
            
            pairs_with_descriptions.append(pair_data)
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs_with_descriptions, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Stored {len(pairs_with_descriptions)} entity pairs in {output_file}")
        
        return output_file

    def get_matched_pairs_files(self, source_language: str = None, target_language: str = None) -> list[str]:
        """
        Get a list of all stored matched pairs JSON files, optionally filtered by language pair.
        
        Args:
            source_language: Optional filter by source language
            target_language: Optional filter by target language
            
        Returns:
            List of file paths to matched pairs JSON files
        """
        import os
        
        output_dir = os.path.join(self.working_dir, "matched_pairs")
        if not os.path.exists(output_dir):
            logger.warning("No matched pairs directory found")
            return []
            
        all_files = [
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.startswith("matched_pairs_") and f.endswith(".json")
        ]
        
        # Filter by language if specified
        if source_language and target_language:
            filtered_files = [
                f for f in all_files 
                if f"matched_pairs_{source_language}_{target_language}_" in os.path.basename(f)
            ]
            return filtered_files
        elif source_language:
            filtered_files = [
                f for f in all_files 
                if f"matched_pairs_{source_language}_" in os.path.basename(f)
            ]
            return filtered_files
        elif target_language:
            filtered_files = [
                f for f in all_files 
                if f"_{target_language}_" in os.path.basename(f)
            ]
            return filtered_files
        
        return sorted(all_files)
        
    def load_matched_pairs(self, file_path: str) -> list[dict]:
        """
        Load matched pairs data from a JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of matched pairs with descriptions
        """
        import json
        import os
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return []
            
    def get_matched_entity_pairs(
        self, 
        source_language: str = None, 
        target_language: str = None, 
        latest_only: bool = True
    ) -> list[dict]:
        """
        Convenience method to retrieve matched entity pairs
        
        Args:
            source_language: Optional filter by source language
            target_language: Optional filter by target language
            latest_only: If True, returns only pairs from the most recent file,
                         otherwise returns all pairs from all matching files
            
        Returns:
            List of matched pairs with descriptions
        """
        matched_files = self.get_matched_pairs_files(
            source_language=source_language,
            target_language=target_language
        )
        
        if not matched_files:
            logger.warning(f"No matched pairs files found for the specified languages")
            return []
            
        if latest_only:
            # Get only the most recent file
            latest_file = sorted(matched_files)[-1]
            logger.info(f"Loading matched pairs from latest file: {os.path.basename(latest_file)}")
            return self.load_matched_pairs(latest_file)
        else:
            # Combine pairs from all matching files
            all_pairs = []
            for file_path in matched_files:
                pairs = self.load_matched_pairs(file_path)
                all_pairs.extend(pairs)
            
            logger.info(f"Loaded {len(all_pairs)} pairs from {len(matched_files)} files")
            return all_pairs

    async def _match_entities_llm(
        self,
        source_entities: list[str],
        target_entities: list[str], 
        source_language: str,
        target_language: str
    ) -> list[tuple[str, str]]:
        """Match entities using LLM"""
        matches = []
        
        for source_entity in source_entities:
            source_name = source_entity.strip('"')
            prompt = f"""
            I need to match equivalent entities across languages.
            Source entity (in {source_language}): {source_name}
            
            Target entities (in {target_language}):
            {', '.join([t.strip('"') for t in target_entities])}
            
            If the source entity has an equivalent in the target entities list, return only that entity name.
            If there's no equivalent, return "NO MATCH".
            """
            
            try:
                response = await self.llm_model_func(prompt)
                response = response.strip()
                
                for target_entity in target_entities:
                    target_name = target_entity.strip('"')
                    if target_name.upper() in response.upper() and "NO MATCH" not in response.upper():
                        matches.append((source_entity, target_entity))
                        logger.info(f"LLM matched: {source_name} with {target_name}")
                        break
            except Exception as e:
                logger.error(f"Error using LLM for entity matching: {e}")
                
        return matches
    async def _insert_done(self) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.text_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

    # async def _add_cross_lingual_links(
    #     self, 
    #     entity_pairs: list[tuple[str, str]],
    #     source_language: str,
    #     target_language: str
    # ) -> None:
    #     """
    #     Add cross-lingual links between matched entity pairs
        
    #     Args:
    #         entity_pairs: List of matched entity pairs (source_entity, target_entity)
    #         source_language: Source language name
    #         target_language: Target language name
    #     """
    #     for source_entity, target_entity in entity_pairs:
    #         # Check if entities still exist (they might have been deleted)
    #         if (await self.chunk_entity_relation_graph.has_node(source_entity) and 
    #             await self.chunk_entity_relation_graph.has_node(target_entity)):
                
    #             # Check if they already have a TRANSLATED relationship
    #             if await self.chunk_entity_relation_graph.has_edge(source_entity, target_entity):
    #                 edge_data = await self.chunk_entity_relation_graph.get_edge(source_entity, target_entity)
    #                 if edge_data and edge_data.get("relation_type") == "translation_equivalent":
    #                     logger.info(f"Entities {source_entity} and {target_entity} already have a translation relationship")
    #                     continue
                
    #             # Create cross-lingual edges based on the pattern seen in the example GraphML
    #             edge_data = {
    #                 "weight": 1.0,
    #                 "description": f"Translation equivalent ({source_language} ↔ {target_language})",
    #                 "keywords": f"translation,cross-lingual,{source_language},{target_language}",
    #                 "relation_type": "translation_equivalent",
    #                 "languages": f"{source_language},{target_language}",
    #                 "source_id": "cross_lingual",
    #                 "original_language": source_language,
    #                 "translated_language": target_language
    #             }
                
    #             # Add edges in both directions with the same data
    #             await self.chunk_entity_relation_graph.upsert_edge(
    #                 source_entity, target_entity, edge_data
    #             )
                
    #             logger.info(f"Created cross-lingual link: {source_entity} <-> {target_entity}")

    async def _add_cross_lingual_links(
        self, 
        entity_pairs: list[tuple[str, str]],
        source_language: str,
        target_language: str
    ) -> None:
        """
        Add cross-lingual links between matched entity pairs
        
        Args:
            entity_pairs: List of matched entity pairs (source_entity, target_entity)
            source_language: Source language name
            target_language: Target language name
        """
        for source_entity, target_entity in entity_pairs:
            # Check if entities still exist (they might have been deleted)
            if (await self.chunk_entity_relation_graph.has_node(source_entity) and 
                await self.chunk_entity_relation_graph.has_node(target_entity)):
                
                # Cải tiến 6: Kiểm tra trùng lặp trước khi tạo liên kết
                if await self.chunk_entity_relation_graph.has_edge(source_entity, target_entity):
                    edge_data = await self.chunk_entity_relation_graph.get_edge(source_entity, target_entity)
                    if edge_data and edge_data.get("relation_type") == "translation_equivalent":
                        # Cải tiến 5: Thêm logging chi tiết
                        logger.info(f"Entities {source_entity} and {target_entity} already have a translation relationship")
                        continue
                
                # Cải tiến 4: Edge data với metadata phong phú hơn
                edge_data = {
                    "weight": 1.0,
                    "description": f"Translation equivalent ({source_language} ↔ {target_language})",
                    "keywords": f"translation,cross-lingual,{source_language},{target_language}",
                    "relation_type": "translation_equivalent",
                    "languages": f"{source_language},{target_language}",
                    "source_id": "cross_lingual",
                    "original_language": source_language,
                    "translated_language": target_language,
                    "created_at": datetime.now().isoformat(),  # Thêm timestamp
                    "confidence_score": 1.0,  # Có thể thay đổi dựa trên phương pháp matching
                    "matching_method": self.addon_params.get("matching_method", "hybrid")  # Lưu phương pháp matching đã sử dụng
                }
                
                # Add edges in both directions with the same data
                try:
                    await self.chunk_entity_relation_graph.upsert_edge(
                        source_entity, target_entity, edge_data
                    )
                    # Cải tiến 5: Thêm logging chi tiết
                    logger.info(f"Created cross-lingual link: {source_entity} <-> {target_entity}")
                    logger.debug(f"Edge data: {edge_data}")
                except Exception as e:
                    # Cải tiến 5: Log lỗi chi tiết
                    logger.error(f"Failed to create cross-lingual link between {source_entity} and {target_entity}: {e}")
                    continue

                # Cập nhật metadata của nodes để đánh dấu chúng đã được liên kết
                try:
                    # Update source node
                    source_node_data = await self.chunk_entity_relation_graph.get_node(source_entity)
                    if source_node_data:
                        source_node_data["has_translations"] = True
                        source_node_data["translated_languages"] = source_node_data.get("translated_languages", "") + f"|{target_language}"
                        await self.chunk_entity_relation_graph.upsert_node(source_entity, source_node_data)

                    # Update target node
                    target_node_data = await self.chunk_entity_relation_graph.get_node(target_entity)
                    if target_node_data:
                        target_node_data["has_translations"] = True
                        target_node_data["translated_languages"] = target_node_data.get("translated_languages", "") + f"|{source_language}"
                        await self.chunk_entity_relation_graph.upsert_node(target_entity, target_node_data)
                    
                    # Cải tiến 5: Log thành công
                    logger.debug(f"Updated metadata for nodes {source_entity} and {target_entity}")
                except Exception as e:
                    # Cải tiến 5: Log lỗi chi tiết
                    logger.error(f"Failed to update node metadata: {e}")

    def insert_custom_kg(self, custom_kg: dict[str, Any]) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict[str, Any]) -> None:
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
            for chunk_data in custom_kg.get("chunks", {}):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data: list[dict[str, str]] = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Detect language from entity name
                language = entity_data.get("language", detect_language(entity_name))

                # Prepare node data
                node_data: dict[str, str] = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                    "language": language,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data: list[dict[str, str]] = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Detect language from description
                language = relationship_data.get("language", detect_language(description))

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                                "language": language,
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                        "language": language,
                    },
                )
                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "language": language,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
                    "language": dp.get("language", "Unknown"),
                }
                for dp in all_entities_data
            }
            await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "content": dp["keywords"]
                    + dp["src_id"]
                    + dp["tgt_id"]
                    + dp["description"],
                    "language": dp.get("language", "Unknown"),
                }
                for dp in all_relationships_data
            }
            await self.relationships_vdb.upsert(data_for_vdb)

        finally:
            if update_storage:
                await self._insert_done()

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, system_prompt))  # type: ignore

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                system_prompt=system_prompt,
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response


    def retrieval(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aretrieval(query, param, system_prompt))  # type: ignore

    async def aretrieval(
        self,
        query: str,
        param: QueryParam = QueryParam(only_need_context=True),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_retrieval(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unsupport mode {param.mode}")
        await self._query_done()
        return response


    def query_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ):
        """
        1. Extract keywords from the 'query' using new function in operate.py.
        2. Then run the standard aquery() flow with the final prompt (formatted_question).
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_separate_keyword_extraction(query, prompt, param)
        )

    async def aquery_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ) -> str | AsyncIterator[str]:
        """
        1. Calls extract_keywords_only to get HL/LL keywords from 'query'.
        2. Then calls kg_query(...) or naive_query(...), etc. as the main query, while also injecting the newly extracted keywords if needed.
        """
        # ---------------------
        # STEP 1: Keyword Extraction
        # ---------------------
        # We'll assume 'extract_keywords_only(...)' returns (hl_keywords, ll_keywords).
        hl_keywords, ll_keywords = await extract_keywords_only(
            text=query,
            param=param,
            global_config=asdict(self),
            hashing_kv=self.llm_response_cache
            or self.key_string_value_json_storage_cls(
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            ),
        )

        param.hl_keywords = hl_keywords
        param.ll_keywords = ll_keywords

        # ---------------------
        # STEP 2: Final Query Logic
        # ---------------------

        # Create a new string with the prompt and the keywords
        ll_keywords_str = ", ".join(ll_keywords)
        hl_keywords_str = ", ".join(hl_keywords)
        formatted_question = f"{prompt}\n\n### Keywords:\nHigh-level: {hl_keywords_str}\nLow-level: {ll_keywords_str}\n\n### Query:\n{query}"

        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query_with_keywords(
                formatted_question,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        elif param.mode == "naive":
            response = await naive_query(
                formatted_question,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                formatted_question,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")

        await self._query_done()
        return response

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    def delete_by_entity(self, entity_name: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str) -> None:
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_entity_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self) -> None:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    def _get_content_summary(self, content: str, max_length: int = 100) -> str:
        """Get summary of document content

        Args:
            content: Original document content
            max_length: Maximum length of summary

        Returns:
            Truncated content with ellipsis if needed
        """
        content = content.strip()
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            # 1. Get the document status and related data
            doc_status = await self.doc_status.get_by_id(doc_id)
            if not doc_status:
                logger.warning(f"Document {doc_id} not found")
                return

            logger.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all related chunks
            chunks = await self.text_chunks.get_by_id(doc_id)
            if not chunks:
                return

            chunk_ids = list(chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # 3. Before deleting, check the related entities and relationships for these chunks
            for chunk_id in chunk_ids:
                # Check entities
                entities = [
                    dp
                    for dp in self.entities_vdb.client_storage["data"]
                    if dp.get("source_id") == chunk_id
                ]
                logger.debug(f"Chunk {chunk_id} has {len(entities)} related entities")

                # Check relationships
                relations = [
                    dp
                    for dp in self.relationships_vdb.client_storage["data"]
                    if dp.get("source_id") == chunk_id
                ]
                logger.debug(f"Chunk {chunk_id} has {len(relations)} related relations")

            # Continue with the original deletion process...

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 5. Find and process entities and relationships that have these chunks as source
            # Get all nodes in the graph
            nodes = self.chunk_entity_relation_graph._graph.nodes(data=True)
            edges = self.chunk_entity_relation_graph._graph.edges(data=True)

            # Track which entities and relationships need to be deleted or updated
            entities_to_delete = set()
            entities_to_update = {}  # entity_name -> new_source_id
            relationships_to_delete = set()
            relationships_to_update = {}  # (src, tgt) -> new_source_id

            # Process entities
            for node, data in nodes:
                if "source_id" in data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        entities_to_delete.add(node)
                        logger.debug(
                            f"Entity {node} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        entities_to_update[node] = new_source_id
                        logger.debug(
                            f"Entity {node} will be updated with new source_id: {new_source_id}"
                        )

            # Process relationships
            for src, tgt, data in edges:
                if "source_id" in data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        relationships_to_delete.add((src, tgt))
                        logger.debug(
                            f"Relationship {src}-{tgt} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        relationships_to_update[(src, tgt)] = new_source_id
                        logger.debug(
                            f"Relationship {src}-{tgt} will be updated with new source_id: {new_source_id}"
                        )

            # Delete entities
            if entities_to_delete:
                for entity in entities_to_delete:
                    await self.entities_vdb.delete_entity(entity)
                    logger.debug(f"Deleted entity {entity} from vector DB")
                self.chunk_entity_relation_graph.remove_nodes(list(entities_to_delete))
                logger.debug(f"Deleted {len(entities_to_delete)} entities from graph")

            # Update entities
            for entity, new_source_id in entities_to_update.items():
                node_data = self.chunk_entity_relation_graph._graph.nodes[entity]
                node_data["source_id"] = new_source_id
                await self.chunk_entity_relation_graph.upsert_node(entity, node_data)
                logger.debug(
                    f"Updated entity {entity} with new source_id: {new_source_id}"
                )

            # Delete relationships
            if relationships_to_delete:
                for src, tgt in relationships_to_delete:
                    rel_id_0 = compute_mdhash_id(src + tgt, prefix="rel-")
                    rel_id_1 = compute_mdhash_id(tgt + src, prefix="rel-")
                    await self.relationships_vdb.delete([rel_id_0, rel_id_1])
                    logger.debug(f"Deleted relationship {src}-{tgt} from vector DB")
                self.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )
                logger.debug(
                    f"Deleted {len(relationships_to_delete)} relationships from graph"
                )

            # Update relationships
            for (src, tgt), new_source_id in relationships_to_update.items():
                edge_data = self.chunk_entity_relation_graph._graph.edges[src, tgt]
                edge_data["source_id"] = new_source_id
                await self.chunk_entity_relation_graph.upsert_edge(src, tgt, edge_data)
                logger.debug(
                    f"Updated relationship {src}-{tgt} with new source_id: {new_source_id}"
                )

            # 6. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            # 7. Ensure all indexes are updated
            await self._insert_done()

            logger.info(
                f"Successfully deleted document {doc_id} and related data. "
                f"Deleted {len(entities_to_delete)} entities and {len(relationships_to_delete)} relationships. "
                f"Updated {len(entities_to_update)} entities and {len(relationships_to_update)} relationships."
            )

            # Add verification step
            async def verify_deletion():
                # Verify if the document has been deleted
                if await self.full_docs.get_by_id(doc_id):
                    logger.error(f"Document {doc_id} still exists in full_docs")

                # Verify if chunks have been deleted
                remaining_chunks = await self.text_chunks.get_by_id(doc_id)
                if remaining_chunks:
                    logger.error(f"Found {len(remaining_chunks)} remaining chunks")

                # Verify entities and relationships
                for chunk_id in chunk_ids:
                    # Check entities
                    entities_with_chunk = [
                        dp
                        for dp in self.entities_vdb.client_storage["data"]
                        if chunk_id
                        in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                    ]
                    if entities_with_chunk:
                        logger.error(
                            f"Found {len(entities_with_chunk)} entities still referencing chunk {chunk_id}"
                        )

                    # Check relationships
                    relations_with_chunk = [
                        dp
                        for dp in self.relationships_vdb.client_storage["data"]
                        if chunk_id
                        in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                    ]
                    if relations_with_chunk:
                        logger.error(
                            f"Found {len(relations_with_chunk)} relations still referencing chunk {chunk_id}"
                        )

            await verify_deletion()

        except Exception as e:
            logger.error(f"Error while deleting document {doc_id}: {e}")

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity

        Args:
            entity_name: Entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing entity information, including:
                - entity_name: Entity name
                - source_id: Source document ID
                - graph_data: Complete node data from the graph database
                - vector_data: (optional) Data from the vector database
        """
        entity_name = f'"{entity_name.upper()}"'

        # Get information from the graph
        node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        result: dict[str, str | None | dict[str, str]] = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": node_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = self.entities_vdb._client.get([entity_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship

        Args:
            src_entity: Source entity name (no need for quotes)
            tgt_entity: Target entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing relationship information, including:
                - src_entity: Source entity name
                - tgt_entity: Target entity name
                - source_id: Source document ID
                - graph_data: Complete edge data from the graph database
                - vector_data: (optional) Data from the vector database
        """
        src_entity = f'"{src_entity.upper()}"'
        tgt_entity = f'"{tgt_entity.upper()}"'

        # Get information from the graph
        edge_data = await self.chunk_entity_relation_graph.get_edge(
            src_entity, tgt_entity
        )
        source_id = edge_data.get("source_id") if edge_data else None

        result: dict[str, str | None | dict[str, str]] = {
            "src_entity": src_entity,
            "tgt_entity": tgt_entity,
            "source_id": source_id,
            "graph_data": edge_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
            vector_data = self.relationships_vdb._client.get([rel_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result

    def set_language(self, language: str) -> None:
        """Set the language for prompts
        
        Args:
            language: Language code ('EN' for English, 'VI' for Vietnamese)
        """
        if language not in ["English", "Vietnamese"]:
            logger.warning(f"Unsupported language: {language}. Falling back to English.")
            language = "EN"
        
        self.addon_params["language"] = language
        logger.info(f"Language set to: {language}")

# new_func
    def insert_duo(
        self,
        data_original,
        data_translated=None,
        source_language="Vietnamese",
        target_language="English",
        store_translations=True,
        translation_db_path=None
    ):
        """
        Insert a document in both its original language and translated version.
        
        Args:
            data_original: Original document data (Vietnamese)
            data_translated: Translated document data (if None, will be generated using LLM)
            source_language: Source language (default: "Vietnamese")
            target_language: Target language (default: "English")
            store_translations: Whether to store entity and relation translations
            translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
        
        Returns:
            Tuple of (original_doc_id, translated_doc_id)
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert_duo(data_original, data_translated, source_language, target_language, store_translations, translation_db_path)
        )

    async def ainsert_duo(
        self,
        data_original,
        data_translated=None,
        source_language="Vietnamese",
        target_language="English",
        store_translations=True,
        translation_db_path=None
    ):
        """
        Async insert a document in both its original language and translated version.
        Ensures exact 1:1 mapping between entities and relationships.
        
        Args:
            data_original: Original document data (Vietnamese)
            data_translated: Translated document data (if None, will be generated using LLM)
            source_language: Source language (default: "Vietnamese")
            target_language: Target language (default: "English")
            store_translations: Whether to store entity and relation translations
            translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
        
        Returns:
            Tuple of (original_doc_id, translated_doc_id)
        """
        if translation_db_path is None:
            translation_db_path = os.path.join(self.working_dir, "translations.json")
        
        logger.info(f"Starting duo insertion: {source_language} and {target_language}")
        
        # First process the original document
        original_doc_id = compute_mdhash_id(data_original.strip(), prefix="doc-")
        
        # Store original document in doc status first
        await self.doc_status.upsert({
            original_doc_id: {
                "content": data_original,
                "content_summary": self._get_content_summary(data_original),
                "content_length": len(data_original),
                "status": DocStatus.PENDING,
                "language": source_language,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        })
        
        # Store data to full_docs
        await self.full_docs.upsert({original_doc_id: {"content": data_original.strip()}})
        
        # Create chunks for original document
        original_chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": original_doc_id,
            }
            for dp in self.chunking_func(
                data_original,
                None,
                False,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name,
            )
        }
        
        # Process the chunks and extract entities/relations
        # Insert chunks to vector storage and text chunks storage
        await asyncio.gather(
            self.chunks_vdb.upsert(original_chunks),
            self.text_chunks.upsert(original_chunks),
        )
        
        # Extract entities and relations from original document
        logger.info(f"Extracting entities and relations from {source_language} document")
        original_extraction_result = await extract_entities(
            original_chunks,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            llm_response_cache=self.llm_response_cache,
            global_config=asdict(self),
        )
        
        # Get entities and relations from the original document
        original_entities = await self._get_document_entities(original_doc_id, original_chunks)
        original_relations = await self._get_document_relations(original_doc_id, original_chunks)
        
        logger.info(f"Found {len(original_entities)} entities and {len(original_relations)} relations in {source_language} document")
        
        # Update status for original document
        await self.doc_status.upsert({
            original_doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_count": len(original_chunks),
                "content": data_original,
                "content_summary": self._get_content_summary(data_original),
                "content_length": len(data_original),
                "updated_at": datetime.now().isoformat(),
            }
        })
        
        # If translated data is not provided, generate it
        if data_translated is None or data_translated == "":
            logger.info(f"Generating translation for document in {target_language}")
            data_translated = await self._translate_preserving_structure(
                data_original, 
                source_language,
                target_language
            )
        
        # Process translated document
        translated_doc_id = compute_mdhash_id(data_translated.strip(), prefix="doc-")
        
        # Store translated document in doc status
        await self.doc_status.upsert({
            translated_doc_id: {
                "content": data_translated,
                "content_summary": self._get_content_summary(data_translated),
                "content_length": len(data_translated),
                "status": DocStatus.PENDING,
                "language": target_language,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        })
        
        # Store data for translated document
        await self.full_docs.upsert({translated_doc_id: {"content": data_translated.strip()}})
        
        # Create chunks for translated document
        translated_chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": translated_doc_id,
            }
            for dp in self.chunking_func(
                data_translated,
                None,
                False,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name,
            )
        }
        
        # Insert chunks to vector storage and text chunks storage
        await asyncio.gather(
            self.chunks_vdb.upsert(translated_chunks),
            self.text_chunks.upsert(translated_chunks),
        )
        
        # Extract corresponding entities and relations in the translated document
        # using the original entities as a guide
        logger.info(f"Extracting matching entities and relations from {target_language} document")
        
        # First pass: Extract matching entities - ONLY EXTRACT, DON'T SAVE YET
        translated_entities = await self._extract_matching_entities(
            original_entities,
            data_translated,
            source_language,
            target_language,
            translated_chunks
        )
        
        # Second pass: Extract matching relations - ONLY EXTRACT, DON'T SAVE YET
        translated_relations = await self._extract_matching_relations(
            original_relations,
            translated_entities,
            data_translated,
            source_language,
            target_language,
            translated_chunks
        )
        
        logger.info(f"Extracted {len(translated_entities)} entities and {len(translated_relations)} relations in {target_language} document")
        
        # Verify the counts match
        if len(original_entities) != len(translated_entities):
            logger.warning(f"Entity count mismatch: {len(original_entities)} {source_language} vs {len(translated_entities)} {target_language}")
            # Force entity count to match by requesting a fix
            translated_entities = await self._fix_entity_count_mismatch(
                original_entities,
                translated_entities,
                data_translated,
                source_language,
                target_language,
                translated_chunks
            )
        
        if len(original_relations) != len(translated_relations):
            logger.warning(f"Relation count mismatch: {len(original_relations)} {source_language} vs {len(translated_relations)} {target_language}")
            # Force relation count to match by requesting a fix
            translated_relations = await self._fix_relation_count_mismatch(
                original_relations,
                translated_relations, 
                translated_entities,
                data_translated,
                source_language, 
                target_language,
                translated_chunks
            )
        
        # NOW THAT WE HAVE VERIFIED ENTITIES AND RELATIONS MATCH, SAVE THEM TO THE GRAPH
        logger.info(f"Saving verified entities and relations to knowledge graph")
        
        # Save entities to graph and vector DB
        for entity in translated_entities:
            entity_name = f'"{entity["name"].upper()}"'
            
            # Get first chunk ID for this document
            chunk_id = next(iter(translated_chunks.keys()))
            
            # Create node data
            node_data = {
                "entity_type": f'"{entity["type"].upper()}"',
                "description": entity["description"],
                "source_id": chunk_id,
                "language": target_language,  # Add language metadata
            }
            
            # Add to knowledge graph
            await self.chunk_entity_relation_graph.upsert_node(entity_name, node_data)
            
            # Add to vector database
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            await self.entities_vdb.upsert({
                entity_id: {
                    "content": f"{entity_name} {entity['description']}",
                    "entity_name": entity_name,
                    "language": target_language,  # Add language metadata
                }
            })
        
        # Save relations to graph and vector DB
        for relation in translated_relations:
            src_entity = f'"{relation["source"].upper()}"'
            tgt_entity = f'"{relation["target"].upper()}"'
            
            # Get first chunk ID for this document
            chunk_id = next(iter(translated_chunks.keys()))
            
            # Create edge data
            edge_data = {
                "description": relation["description"],
                "keywords": relation["keywords"],
                "weight": 1.0,
                "source_id": chunk_id,
                "language": target_language,  # Add language metadata
            }
            
            # Ensure both nodes exist
            for entity in [src_entity, tgt_entity]:
                if not await self.chunk_entity_relation_graph.has_node(entity):
                    # Create placeholder node
                    placeholder_data = {
                        "entity_type": '"UNKNOWN"',
                        "description": "Auto-created entity for relation",
                        "source_id": chunk_id,
                        "language": target_language,  # Add language metadata
                    }
                    await self.chunk_entity_relation_graph.upsert_node(entity, placeholder_data)
            
            # Add edge to knowledge graph
            await self.chunk_entity_relation_graph.upsert_edge(src_entity, tgt_entity, edge_data)
            
            # Add to vector database
            relation_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
            await self.relationships_vdb.upsert({
                relation_id: {
                    "content": f"{relation['keywords']} {src_entity} {tgt_entity} {relation['description']}",
                    "src_id": src_entity,
                    "tgt_id": tgt_entity,
                    "language": target_language,  # Add language metadata
                }
            })
        
        # Update status for translated document
        await self.doc_status.upsert({
            translated_doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_count": len(translated_chunks),
                "content": data_translated,
                "content_summary": self._get_content_summary(data_translated),
                "content_length": len(data_translated),
                "updated_at": datetime.now().isoformat(),
            }
        })
        
        # Create cross-lingual edges between entities
        logger.info(f"Creating cross-lingual edges between {source_language} and {target_language} entities")
        await self._create_cross_lingual_edges(
            original_entities,
            translated_entities,
            source_language,
            target_language
        )
        
        # Store translation pairs if requested
        if store_translations:
            logger.info(f"Storing translation pairs to {translation_db_path}")
            await self._store_translation_pairs(
                original_entities,
                translated_entities,
                original_relations,
                translated_relations,
                source_language,
                target_language,
                translation_db_path
            )
        
        # Save changes to all storages
        await self._insert_done()
        
        return original_doc_id, translated_doc_id

    async def _get_document_entities(self, doc_id: str, chunks: dict = None) -> list[dict]:
        """
        Get entities associated with a specific document.
        
        Args:
            doc_id: Document ID
            chunks: Optional dictionary of chunks to use for matching
            
        Returns:
            List of entity dictionaries with name, type, and description
        """
        # Get all chunks for this document if not provided
        if chunks is None:
            doc_chunks = {}
            chunks_data = await self.text_chunks.get_all()
            for chunk_id, chunk_data in chunks_data.items():
                if chunk_data.get('full_doc_id') == doc_id:
                    doc_chunks[chunk_id] = chunk_data
        else:
            doc_chunks = chunks
        
        if not doc_chunks:
            logger.warning(f"No chunks found for document {doc_id}")
            return []
        
        # Get entities that have these chunks as source
        all_nodes = self.chunk_entity_relation_graph._graph.nodes(data=True)
        doc_entities = []
        
        for node_id, data in all_nodes:
            if 'source_id' in data:
                # Check if any source is from the document's chunks
                sources = data['source_id'].split(GRAPH_FIELD_SEP)
                for chunk_id in doc_chunks:
                    if chunk_id in sources:
                        # Add entity with its metadata
                        entity_data = {
                            "name": node_id.strip('"'),
                            "type": data.get('entity_type', '').strip('"'),
                            "description": data.get('description', '')
                        }
                        doc_entities.append(entity_data)
                        break
        
        return doc_entities

    async def _get_document_relations(self, doc_id: str, chunks: dict = None) -> list[dict]:
        """
        Get relations associated with a specific document.
        
        Args:
            doc_id: Document ID
            chunks: Optional dictionary of chunks to use for matching
            
        Returns:
            List of relation dictionaries with source, target, and description
        """
        # Get all chunks for this document if not provided
        if chunks is None:
            doc_chunks = {}
            chunks_data = await self.text_chunks.get_all()
            for chunk_id, chunk_data in chunks_data.items():
                if chunk_data.get('full_doc_id') == doc_id:
                    doc_chunks[chunk_id] = chunk_data
        else:
            doc_chunks = chunks
        
        if not doc_chunks:
            logger.warning(f"No chunks found for document {doc_id}")
            return []
        
        # Get relations that have these chunks as source
        all_edges = self.chunk_entity_relation_graph._graph.edges(data=True)
        doc_relations = []
        
        for src, tgt, data in all_edges:
            if 'source_id' in data:
                # Check if any source is from the document's chunks
                sources = data['source_id'].split(GRAPH_FIELD_SEP)
                for chunk_id in doc_chunks:
                    if chunk_id in sources:
                        # Add relation with its metadata
                        relation_data = {
                            "source": src.strip('"'),
                            "target": tgt.strip('"'),
                            "description": data.get('description', ''),
                            "keywords": data.get('keywords', '')
                        }
                        doc_relations.append(relation_data)
                        break
        
        return doc_relations

    async def _translate_preserving_structure(
        self, content: str, source_language: str, target_language: str
    ) -> str:
        """
        Translate content while preserving semantic structure.
        
        Args:
            content: Content to translate
            source_language: Source language
            target_language: Target language
            
        Returns:
            Translated content
        """
        prompt = f"""
        You are a professional translator specialized in {source_language}-to-{target_language} translation.
        
        Please translate the following {source_language} text into {target_language}.
        Your translation must:
        1. Maintain the exact same meaning and information as the original
        2. Preserve all named entities (people, organizations, locations, etc.)
        3. Keep the same document structure and flow
        4. Sound natural in {target_language}
        
        Original {source_language} text:
        {content}
        
        {target_language} translation:
        """
        
        response = await self.llm_model_func(prompt)
        return response.strip()

    async def _extract_matching_entities(
        self, 
        original_entities: list[dict],
        translated_text: str,
        source_language: str,
        target_language: str,
        translated_chunks: dict
    ) -> list[dict]:
        """
        Extract entities from translated text that correspond to original entities.
        This ensures a 1:1 mapping between source and target entities.
        
        Args:
            original_entities: List of original entities
            translated_text: Translated document text
            source_language: Source language
            target_language: Target language
            translated_chunks: Dictionary of translated chunks
            
        Returns:
            List of entity dictionaries in the target language
        """
        # Format original entities list for the prompt
        formatted_entities = []
        for i, entity in enumerate(original_entities):
            formatted_entities.append(f"{i+1}. {entity['name']} (Type: {entity['type']}): {entity['description']}")
        
        original_entities_str = "\n".join(formatted_entities)
        
        # Create prompt for entity extraction
        prompt = f"""
        You are tasked with finding the exact equivalent entities in a {target_language} text 
        that correspond to entities identified in the original {source_language} text.
        
        I have identified the following entities in the {source_language} text:
        {original_entities_str}
        
        Now I need you to identify the EXACT corresponding entities in this {target_language} text:
        {translated_text}
        
        IMPORTANT REQUIREMENTS:
        1. You MUST identify EXACTLY {len(original_entities)} entities in the {target_language} text.
        2. Each entity must correspond to one entity in the {source_language} list in the same order.
        3. The entity types should remain the same.
        4. If an entity name is a proper noun or name, it may be spelled the same in both languages.
        5. IMPORTANT: You MUST provide the description in {target_language}, not in {source_language}. Even if you see the entity in the text with a description in {source_language}, you need to translate that description to {target_language}.
        
        Format your response EXACTLY as follows, with one entity per line:
        1. Entity: [entity name] | Type: [entity type] | Description: [description in {target_language}]
        2. Entity: [entity name] | Type: [entity type] | Description: [description in {target_language}]
        ...and so on until you have EXACTLY {len(original_entities)} entities.
        
        Think carefully about each entity and ensure you've found its most accurate equivalent.
        Make sure all descriptions are in {target_language}, not in {source_language}.
        """
        
        response = await self.llm_model_func(prompt)
        
        # Parse the response to extract entity information
        translated_entities = []
        entity_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '|' in line]
        
        for line in entity_lines:
            try:
                # Remove any numbering at the start
                line = re.sub(r'^\d+\.\s*', '', line)
                
                # Extract entity details
                entity_parts = line.split('|')
                if len(entity_parts) >= 3:
                    entity_name = entity_parts[0].split('Entity:')[1].strip() if 'Entity:' in entity_parts[0] else entity_parts[0].strip()
                    entity_type = entity_parts[1].split('Type:')[1].strip() if 'Type:' in entity_parts[1] else entity_parts[1].strip()
                    entity_desc = entity_parts[2].split('Description:')[1].strip() if 'Description:' in entity_parts[2] else entity_parts[2].strip()
                    
                    translated_entities.append({
                        "name": entity_name,
                        "type": entity_type,
                        "description": entity_desc
                    })
            except Exception as e:
                logger.warning(f"Error parsing entity line: {line}. Error: {e}")
        
        # Only return the extracted entities, don't save to graph yet
        return translated_entities

    async def _extract_matching_relations(
        self,
        original_relations: list[dict],
        translated_entities: list[dict],
        translated_text: str,
        source_language: str,
        target_language: str,
        translated_chunks: dict
    ) -> list[dict]:
        """
        Extract relations from translated text that correspond to original relations.
        
        Args:
            original_relations: List of original relations
            translated_entities: List of translated entities
            translated_text: Translated document text
            source_language: Source language
            target_language: Target language
            translated_chunks: Dictionary of translated chunks
            
        Returns:
            List of relation dictionaries in the target language
        """
        if not original_relations:
            return []
        
        # Create entity name mapping for easy lookup
        entity_name_map = {entity["name"].upper(): entity["name"] for entity in translated_entities}
        
        # Format original relations list for the prompt
        formatted_relations = []
        for i, relation in enumerate(original_relations):
            formatted_relations.append(f"{i+1}. {relation['source']} → {relation['target']}: {relation['description']}")
        
        original_relations_str = "\n".join(formatted_relations)
        
        # Format translated entities for reference
        formatted_entities = []
        for i, entity in enumerate(translated_entities):
            formatted_entities.append(f"{i+1}. {entity['name']} (Type: {entity['type']})")
        
        translated_entities_str = "\n".join(formatted_entities)
        
        # Create prompt for relation extraction
        prompt = f"""
        You are tasked with finding the exact equivalent relationships in a {target_language} text 
        that correspond to relationships identified in the original {source_language} text.
        
        I have identified the following relationships in the {source_language} text:
        {original_relations_str}
        
        I have already identified these entities in the {target_language} text:
        {translated_entities_str}
        
        Now I need you to identify the EXACT corresponding relationships in this {target_language} text:
        {translated_text}
        
        IMPORTANT REQUIREMENTS:
        1. You MUST identify EXACTLY {len(original_relations)} relationships in the {target_language} text.
        2. Each relationship must correspond to one relationship in the {source_language} list in the same order.
        3. ONLY use entities from the list of {target_language} entities I provided.
        4. The relationships should maintain the same directional meaning (A → B).
        5. IMPORTANT: You MUST provide the description in {target_language}, not in {source_language}. Even if you see the relationship in the text with a description in {source_language}, you need to translate that description to {target_language}.
        
        Format your response EXACTLY as follows, with one relationship per line:
        1. [source entity] → [target entity] | Description: [relationship description in {target_language}] | Keywords: [comma-separated keywords in {target_language}]
        2. [source entity] → [target entity] | Description: [relationship description in {target_language}] | Keywords: [comma-separated keywords in {target_language}]
        ...and so on until you have EXACTLY {len(original_relations)} relationships.
        
        Think carefully about each relationship and ensure you've found its most accurate equivalent.
        Make sure all descriptions and keywords are in {target_language}, not in {source_language}.
        """
        
        response = await self.llm_model_func(prompt)
        
        # Parse the response to extract relation information
        translated_relations = []
        relation_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '→' in line]
        
        for line in relation_lines:
            try:
                # Remove any numbering at the start
                line = re.sub(r'^\d+\.\s*', '', line)
                
                # Split into relationship and metadata
                rel_parts = line.split('|')
                if len(rel_parts) >= 2:
                    # Parse the relationship part (source → target)
                    rel_entities = rel_parts[0].strip().split('→')
                    if len(rel_entities) == 2:
                        src_entity = rel_entities[0].strip()
                        tgt_entity = rel_entities[1].strip()
                        
                        # Parse description and keywords
                        rel_desc = ""
                        rel_keywords = ""
                        
                        for part in rel_parts[1:]:
                            if 'Description:' in part:
                                rel_desc = part.split('Description:')[1].strip()
                            elif 'Keywords:' in part:
                                rel_keywords = part.split('Keywords:')[1].strip()
                        
                        translated_relations.append({
                            "source": src_entity,
                            "target": tgt_entity,
                            "description": rel_desc,
                            "keywords": rel_keywords
                        })
            except Exception as e:
                logger.warning(f"Error parsing relationship line: {line}. Error: {e}")
        
        # Only return the extracted relations, don't save to graph yet
        return translated_relations

    async def _store_translation_pairs(
        self, 
        original_entities: list[dict], 
        translated_entities: list[dict],
        original_relations: list[dict], 
        translated_relations: list[dict],
        source_language: str, 
        target_language: str, 
        db_path: str
    ) -> None:
        """
        Store entity and relation translation pairs in a JSON file for future use.
        
        Args:
            original_entities: Entities from original text
            translated_entities: Entities from translated text
            original_relations: Relations from original text
            translated_relations: Relations from translated text
            source_language: Source language name
            target_language: Target language name
            db_path: Path to save the translation database
        """
        from lightrag.utils import load_json, write_json
        
        # Ensure we have equal numbers of entities and relations
        if len(original_entities) != len(translated_entities):
            # Take the minimum number to ensure pairs
            min_count = min(len(original_entities), len(translated_entities))
            original_entities = original_entities[:min_count]
            translated_entities = translated_entities[:min_count]
            
        if len(original_relations) != len(translated_relations):
            # Take the minimum number to ensure pairs
            min_count = min(len(original_relations), len(translated_relations))
            original_relations = original_relations[:min_count]
            translated_relations = translated_relations[:min_count]
        
        # Create entity translation pairs
        entity_translations = []
        for i in range(len(original_entities)):
            entity_translations.append({
                "original": original_entities[i]["name"],
                "translated": translated_entities[i]["name"],
                "original_description": original_entities[i]["description"],
                "translated_description": translated_entities[i]["description"],
                "original_language": source_language,
                "translated_language": target_language,
                "type": original_entities[i]["type"],
                "timestamp": datetime.now().isoformat()
            })
        
        # Create relation translation pairs
        relation_translations = []
        for i in range(len(original_relations)):
            relation_translations.append({
                "original_src": original_relations[i]["source"],
                "original_tgt": original_relations[i]["target"],
                "original_desc": original_relations[i]["description"],
                "translated_src": translated_relations[i]["source"],
                "translated_tgt": translated_relations[i]["target"],
                "translated_desc": translated_relations[i]["description"],
                "original_language": source_language,
                "translated_language": target_language,
                "timestamp": datetime.now().isoformat()
            })
        
        # Load existing translations if file exists
        translations_db = {}
        try:
            if os.path.exists(db_path):
                translations_db = load_json(db_path) or {}
        except Exception as e:
            logger.warning(f"Could not load existing translations: {e}")
        
        # Add new translations
        if 'entities' not in translations_db:
            translations_db['entities'] = []
        if 'relations' not in translations_db:
            translations_db['relations'] = []
        
        translations_db['entities'].extend(entity_translations)
        translations_db['relations'].extend(relation_translations)
        
        # Remove duplicates
        translations_db['entities'] = self._deduplicate_translations(translations_db['entities'])
        translations_db['relations'] = self._deduplicate_translations(translations_db['relations'])
        
        # Save to file
        write_json(translations_db, db_path)
        logger.info(f"Saved {len(entity_translations)} entity and {len(relation_translations)} relation translations to {db_path}")

    def _deduplicate_translations(self, translations_list: list[dict]) -> list[dict]:
        """
        Remove duplicate translations from a list of translation pairs.
        
        Args:
            translations_list: List of translation dictionaries
            
        Returns:
            Deduplicated list of translation dictionaries
        """
        seen = set()
        unique_translations = []
        
        for trans in translations_list:
            # Use relevant fields to create a unique key
            if 'original' in trans and 'translated' in trans:
                key = f"{trans['original']}|{trans['translated']}|{trans.get('original_language', '')}|{trans.get('translated_language', '')}"
            elif 'original_src' in trans and 'original_tgt' in trans:
                key = (f"{trans['original_src']}|{trans['original_tgt']}|"
                    f"{trans['translated_src']}|{trans['translated_tgt']}|"
                    f"{trans.get('original_language', '')}|{trans.get('translated_language', '')}")
            else:
                # If we can't determine a unique key, keep the entry
                unique_translations.append(trans)
                continue
            
            if key not in seen:
                seen.add(key)
                unique_translations.append(trans)
        
        return unique_translations
    async def _fix_entity_count_mismatch(
        self, 
        original_entities: list[dict],
        translated_entities: list[dict],
        translated_text: str,
        source_language: str,
        target_language: str,
        translated_chunks: dict
    ) -> list[dict]:
        """
        Fix entity count mismatch by requesting additional entities or removing excess ones.
        
        Args:
            original_entities: List of original entities
            translated_entities: List of translated entities (potentially mismatched)
            translated_text: Translated document text
            source_language: Source language
            target_language: Target language
            translated_chunks: Dictionary of translated chunks
            
        Returns:
            Updated list of translated entities with correct count
        """
        orig_count = len(original_entities)
        trans_count = len(translated_entities)
        
        if orig_count == trans_count:
            return translated_entities
        
        # Format original entities
        orig_entities_str = "\n".join([
            f"{i+1}. {entity['name']} (Type: {entity['type']}): {entity['description']}"
            for i, entity in enumerate(original_entities)
        ])
        
        # Format existing translated entities
        trans_entities_str = "\n".join([
            f"{i+1}. {entity['name']} (Type: {entity['type']}): {entity['description']}"
            for i, entity in enumerate(translated_entities)
        ])
        
        # Build the appropriate prompt based on the mismatch type
        if trans_count < orig_count:
            # We need more entities
            prompt = f"""
            I need your help fixing an entity count mismatch between {source_language} and {target_language} versions of a document.
            
            The {source_language} document has {orig_count} entities:
            {orig_entities_str}
            
            But we only identified {trans_count} entities in the {target_language} document:
            {trans_entities_str}
            
            Please identify {orig_count - trans_count} ADDITIONAL entities in the {target_language} text that would complete the list,
            ensuring we have exactly one {target_language} entity for each {source_language} entity.
            
            {target_language} text:
            {translated_text}
            
            Format your response EXACTLY as follows for ONLY the ADDITIONAL entities:
            {trans_count+1}. Entity: [entity name] | Type: [entity type] | Description: [description]
            {trans_count+2}. Entity: [entity name] | Type: [entity type] | Description: [description]
            ...and so on until we have a total of {orig_count} entities.
            """
            
            response = await self.llm_model_func(prompt)
            
            # Parse the additional entities
            additional_entities = []
            entity_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '|' in line]
            
            for line in entity_lines:
                try:
                    # Remove any numbering at the start
                    line = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Extract entity details
                    entity_parts = line.split('|')
                    if len(entity_parts) >= 3:
                        entity_name = entity_parts[0].split('Entity:')[1].strip() if 'Entity:' in entity_parts[0] else entity_parts[0].strip()
                        entity_type = entity_parts[1].split('Type:')[1].strip() if 'Type:' in entity_parts[1] else entity_parts[1].strip()
                        entity_desc = entity_parts[2].split('Description:')[1].strip() if 'Description:' in entity_parts[2] else entity_parts[2].strip()
                        
                        additional_entities.append({
                            "name": entity_name,
                            "type": entity_type,
                            "description": entity_desc
                        })
                except Exception as e:
                    logger.warning(f"Error parsing entity line: {line}. Error: {e}")
            
            # Combine existing and additional entities
            updated_entities = translated_entities + additional_entities
            
        else:
            # We have too many entities - keep only the most important ones
            prompt = f"""
            I need your help fixing an entity count mismatch between {source_language} and {target_language} versions of a document.
            
            The {source_language} document has {orig_count} entities:
            {orig_entities_str}
            
            But we identified {trans_count} entities in the {target_language} document:
            {trans_entities_str}
            
            Please select EXACTLY {orig_count} entities from the {target_language} list that best match the entities in the {source_language} list.
            
            Format your response EXACTLY as follows for the SELECTED entities:
            1. Entity: [entity name] | Type: [entity type] | Description: [description]
            2. Entity: [entity name] | Type: [entity type] | Description: [description]
            ...and so on until you have EXACTLY {orig_count} entities.
            
            Make sure that each selected entity corresponds with one of the original entities in order.
            """
            
            response = await self.llm_model_func(prompt)
            
            # Parse the selected entities
            updated_entities = []
            entity_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '|' in line]
            
            for line in entity_lines:
                try:
                    # Remove any numbering at the start
                    line = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Extract entity details
                    entity_parts = line.split('|')
                    if len(entity_parts) >= 3:
                        entity_name = entity_parts[0].split('Entity:')[1].strip() if 'Entity:' in entity_parts[0] else entity_parts[0].strip()
                        entity_type = entity_parts[1].split('Type:')[1].strip() if 'Type:' in entity_parts[1] else entity_parts[1].strip()
                        entity_desc = entity_parts[2].split('Description:')[1].strip() if 'Description:' in entity_parts[2] else entity_parts[2].strip()
                        
                        updated_entities.append({
                            "name": entity_name,
                            "type": entity_type,
                            "description": entity_desc
                        })
                except Exception as e:
                    logger.warning(f"Error parsing entity line: {line}. Error: {e}")
        
        # Verify the count is correct now
        if len(updated_entities) != orig_count:
            logger.warning(f"Entity count still mismatched after fixing: {len(updated_entities)} vs {orig_count}")
            # Force the count to match by truncating or padding
            if len(updated_entities) > orig_count:
                updated_entities = updated_entities[:orig_count]
            else:
                # Duplicate the last entity if needed
                while len(updated_entities) < orig_count:
                    duplicate = dict(updated_entities[-1])
                    duplicate["name"] = f"{duplicate['name']} (Copy)"
                    updated_entities.append(duplicate)
        
        return updated_entities

    async def _fix_relation_count_mismatch(
        self,
        original_relations: list[dict],
        translated_relations: list[dict], 
        translated_entities: list[dict],
        data_translated: str,
        source_language: str, 
        target_language: str,
        translated_chunks: dict
    ) -> list[dict]:
        """
        Fix relation count mismatch by requesting additional relations or removing excess ones.
        
        Args:
            original_relations: List of original relations
            translated_relations: List of translated relations (potentially mismatched)
            translated_entities: List of translated entities
            data_translated: Translated document text
            source_language: Source language
            target_language: Target language
            translated_chunks: Dictionary of translated chunks
            
        Returns:
            Updated list of translated relations with correct count
        """
        orig_count = len(original_relations)
        trans_count = len(translated_relations)
        
        if orig_count == trans_count:
            return translated_relations
        
        # Format original relations
        orig_relations_str = "\n".join([
            f"{i+1}. {rel['source']} → {rel['target']}: {rel['description']}"
            for i, rel in enumerate(original_relations)
        ])
        
        # Format existing translated relations
        trans_relations_str = "\n".join([
            f"{i+1}. {rel['source']} → {rel['target']}: {rel['description']}"
            for i, rel in enumerate(translated_relations)
        ])
        
        # Format translated entities for reference
        trans_entities_str = "\n".join([
            f"{i+1}. {entity['name']} (Type: {entity['type']})"
            for i, entity in enumerate(translated_entities)
        ])
        
        # Build the appropriate prompt based on the mismatch type
        if trans_count < orig_count:
            # We need more relations
            prompt = f"""
            I need your help fixing a relationship count mismatch between {source_language} and {target_language} versions of a document.
            
            The {source_language} document has {orig_count} relationships:
            {orig_relations_str}
            
            But we only identified {trans_count} relationships in the {target_language} document:
            {trans_relations_str}
            
            Here are the entities available in the {target_language} document:
            {trans_entities_str}
            
            Please identify {orig_count - trans_count} ADDITIONAL relationships in the {target_language} text that would complete the list,
            ensuring we have exactly one {target_language} relationship for each {source_language} relationship.
            
            {target_language} text:
            {data_translated}
            
            Format your response EXACTLY as follows for ONLY the ADDITIONAL relationships:
            {trans_count+1}. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
            {trans_count+2}. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
            ...and so on until we have a total of {orig_count} relationships.
            
            ONLY use entity names from the list I provided above.
            """
            
            response = await self.llm_model_func(prompt)
            
            # Parse the additional relations
            additional_relations = []
            relation_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '→' in line]
            
            for line in relation_lines:
                try:
                    # Remove any numbering at the start
                    line = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Split into relationship and metadata
                    rel_parts = line.split('|')
                    if len(rel_parts) >= 2:
                        # Parse the relationship part (source → target)
                        rel_entities = rel_parts[0].strip().split('→')
                        if len(rel_entities) == 2:
                            src_entity = rel_entities[0].strip()
                            tgt_entity = rel_entities[1].strip()
                            
                            # Parse description and keywords
                            rel_desc = ""
                            rel_keywords = ""
                            
                            for part in rel_parts[1:]:
                                if 'Description:' in part:
                                    rel_desc = part.split('Description:')[1].strip()
                                elif 'Keywords:' in part:
                                    rel_keywords = part.split('Keywords:')[1].strip()
                            
                            additional_relations.append({
                                "source": src_entity,
                                "target": tgt_entity,
                                "description": rel_desc,
                                "keywords": rel_keywords
                            })
                except Exception as e:
                    logger.warning(f"Error parsing relationship line: {line}. Error: {e}")
            
            # Combine existing and additional relations
            updated_relations = translated_relations + additional_relations
            
        else:
            # We have too many relations - keep only the most important ones
            prompt = f"""
            I need your help fixing a relationship count mismatch between {source_language} and {target_language} versions of a document.
            
            The {source_language} document has {orig_count} relationships:
            {orig_relations_str}
            
            But we identified {trans_count} relationships in the {target_language} document:
            {trans_relations_str}
            
            Please select EXACTLY {orig_count} relationships from the {target_language} list that best match the relationships in the {source_language} list.
            
            Format your response EXACTLY as follows for the SELECTED relationships:
            1. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
            2. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
            ...and so on until you have EXACTLY {orig_count} relationships.
            
            Make sure that each selected relationship corresponds with one of the original relationships in order.
            """
            
            response = await self.llm_model_func(prompt)
            
            # Parse the selected relations
            updated_relations = []
            relation_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '→' in line]
            
            for line in relation_lines:
                try:
                    # Remove any numbering at the start
                    line = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Split into relationship and metadata
                    rel_parts = line.split('|')
                    if len(rel_parts) >= 2:
                        # Parse the relationship part (source → target)
                        rel_entities = rel_parts[0].strip().split('→')
                        if len(rel_entities) == 2:
                            src_entity = rel_entities[0].strip()
                            tgt_entity = rel_entities[1].strip()
                            
                            # Parse description and keywords
                            rel_desc = ""
                            rel_keywords = ""
                            
                            for part in rel_parts[1:]:
                                if 'Description:' in part:
                                    rel_desc = part.split('Description:')[1].strip()
                                elif 'Keywords:' in part:
                                    rel_keywords = part.split('Keywords:')[1].strip()
                            
                            updated_relations.append({
                                "source": src_entity,
                                "target": tgt_entity,
                                "description": rel_desc,
                                "keywords": rel_keywords
                            })
                except Exception as e:
                    logger.warning(f"Error parsing relationship line: {line}. Error: {e}")
        
        # Verify the count is correct now
        if len(updated_relations) != orig_count:
            logger.warning(f"Relation count still mismatched after fixing: {len(updated_relations)} vs {orig_count}")
            # Force the count to match by truncating or padding
            if len(updated_relations) > orig_count:
                updated_relations = updated_relations[:orig_count]
            else:
                # Duplicate the last relation if needed
                while len(updated_relations) < orig_count:
                    duplicate = dict(updated_relations[-1])
                    duplicate["description"] = f"{duplicate['description']} (Copy)"
                    updated_relations.append(duplicate)
        
        return updated_relations

    async def _create_cross_lingual_edges(
        self, 
        original_entities: list[dict], 
        translated_entities: list[dict],
        source_language: str, 
        target_language: str
    ) -> None:
        """
        Create edges in the knowledge graph to connect original entities with their translations.
        
        Args:
            original_entities: List of entities from original text
            translated_entities: List of entities from translated text
            source_language: Source language name
            target_language: Target language name
        """
        # Ensure we have equal numbers of entities
        if len(original_entities) != len(translated_entities):
            logger.warning(f"Entity count mismatch: {len(original_entities)} {source_language} vs {len(translated_entities)} {target_language}")
            # Take the minimum number to ensure pairs
            min_count = min(len(original_entities), len(translated_entities))
            original_entities = original_entities[:min_count]
            translated_entities = translated_entities[:min_count]
        
        # Create cross-lingual edges - one edge for each entity pair
        for i in range(len(original_entities)):
            orig_entity = original_entities[i]
            trans_entity = translated_entities[i]
            
            orig_name = f'"{orig_entity["name"].upper()}"'
            trans_name = f'"{trans_entity["name"].upper()}"'
            
            # Create the cross-lingual edge
            edge_data = {
                "weight": 1.0,
                "description": f"Translation equivalent ({source_language} ↔ {target_language})",
                "keywords": f"translation,cross-lingual,{source_language.lower()},{target_language.lower()}",
                "relation_type": "translation_equivalent",
                "languages": f"{source_language},{target_language}",
                "source_id": "cross_lingual",
                "original_language": source_language,
                "translated_language": target_language,
                "created_at": datetime.now().isoformat(),
                "confidence_score": 1.0,
            }
            
            # Add to knowledge graph
            await self.chunk_entity_relation_graph.upsert_edge(orig_name, trans_name, edge_data)
            
            logger.debug(f"Created cross-lingual edge between {orig_name} ({source_language}) and {trans_name} ({target_language})")
            
        logger.info(f"Created {len(original_entities)} cross-lingual edges between {source_language} and {target_language} entities")
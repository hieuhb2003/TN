from __future__ import annotations

import asyncio
import os
import configparser
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterator, cast
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
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(input, split_by_character, split_by_character_only)
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """
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

        except Exception as e:
            logger.error("Failed to extract entities and relationships")
            raise e

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

                # Prepare node data
                node_data: dict[str, str] = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
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
                    },
                )
                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
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

#### new_func


    def insert_duo(
        self,
        data_original,
        data_translated=None,
        source_language: str,
        target_language: str,
        store_translations=True,
        translation_db_path=None
    ):
        """
        Insert a document in both its original language and translated version.
        
        Args:
            data_original: Original document data
            data_translated: Translated document data (if None, will be generated using LLM)
            target_language: Target language for translation if data_translated is None
            store_translations: Whether to store entity and relation translations
            translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
        
        Returns:
            Tuple of (original_doc_id, translated_doc_id)
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert_duo(data_original, data_translated, target_language, store_translations, translation_db_path)
        )

    async def ainsert_duo(
        self,
        data_original,
        data_translated=None,
        source_language: str,
        target_language="Vietnamese",
        store_translations=True,
        translation_db_path=None
    ):
        """
        Async insert a document in both its original language and translated version.
        
        Args:
            data_original: Original document data
            data_translated: Translated document data (if None, will be generated using LLM)
            target_language: Target language for translation if data_translated is None
            store_translations: Whether to store entity and relation translations
            translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
        
        Returns:
            Tuple of (original_doc_id, translated_doc_id)
        """
        if translation_db_path is None:
            translation_db_path = os.path.join(self.working_dir, "translations.json")
        
        # First process the original document
        logger.info(f"Processing original document in duo insertion")
        original_doc_id = compute_mdhash_id(data_original.strip(), prefix="doc-")
        
        # Store original document in doc status first
        await self.doc_status.upsert({
            original_doc_id: {
                "content": data_original,
                "content_summary": self._get_content_summary(data_original),
                "content_length": len(data_original),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        })
        
        # Store data and get entities/relations
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
        
        # Create a tracker for original entities and relations
        self._last_inserted_doc_id = original_doc_id
        self._original_entities = []
        self._original_relations = []
        
        # Process the chunks and extract entities/relations
        await asyncio.gather(
            self.chunks_vdb.upsert(original_chunks),
            self._process_entity_relation_graph_with_tracking(original_chunks),
            self.text_chunks.upsert(original_chunks),
        )
        
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
        
        # If translated data is not provided, use LLM to translate while preserving entities and relations
        if data_translated is None or data_translated == "":
            logger.info(f"Generating translation for document in {target_language}")
            data_translated = await self._translate_preserving_structure(
                data_original, 
                target_language, 
                self._original_entities,
                self._original_relations
            )
        
        # Process translated document
        logger.info(f"Processing translated document in duo insertion")
        translated_doc_id = compute_mdhash_id(data_translated.strip(), prefix="doc-")
        
        # Store translated document in doc status
        await self.doc_status.upsert({
            translated_doc_id: {
                "content": data_translated,
                "content_summary": self._get_content_summary(data_translated),
                "content_length": len(data_translated),
                "status": DocStatus.PENDING,
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
        
        # Create a tracker for translated entities and relations
        self._last_inserted_doc_id = translated_doc_id
        self._translated_entities = []
        self._translated_relations = []
        
        # Process the translated chunks using the guided extraction with original entities
        await asyncio.gather(
            self.chunks_vdb.upsert(translated_chunks),
            self._process_entity_relation_graph_with_tracking(
                translated_chunks, 
                original_entities=self._original_entities,
                original_relations=self._original_relations
            ),
            self.text_chunks.upsert(translated_chunks),
        )
        
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
        if len(self._original_entities) > 0 and len(self._translated_entities) > 0:
            await self._create_cross_lingual_edges(
                self._original_entities,
                self._translated_entities,
                source_language=source_language,
                target_language=target_language
            )
        
        # Store translation pairs if requested
        if store_translations and len(self._original_entities) > 0 and len(self._translated_entities) > 0:
            await self._store_translation_pairs(
                self._original_entities,
                self._translated_entities,
                self._original_relations,
                self._translated_relations,
                target_language,
                translation_db_path
            )
        
        # Save changes to all storages
        await self._insert_done()
        
        # Clean up tracking variables
        self._last_inserted_doc_id = None
        self._original_entities = []
        self._original_relations = []
        self._translated_entities = []
        self._translated_relations = []
        
        return original_doc_id, translated_doc_id


    async def _process_entity_relation_graph_with_tracking(self, chunk: dict[str, Any], original_entities=None, original_relations=None) -> None:
        """
        Process entity and relation extraction while tracking the entities and relations
        for cross-lingual alignment.
        
        Args:
            chunk: The text chunks to process
            original_entities: Optional list of entities from original document to guide extraction
            original_relations: Optional list of relations from original document to guide extraction
        """
        try:
            # If we have original entities/relations and this is for translated document
            if original_entities and original_relations and hasattr(self, '_last_inserted_doc_id'):
                if hasattr(self, '_original_entities') and self._original_entities and hasattr(self, '_translated_entities'):
                    # This is for translated document - use guided extraction
                    extraction_result = await self._extract_entities_guided(
                        chunk,
                        original_entities,
                        original_relations
                    )
                else:
                    # Standard extraction for original document
                    extraction_result = await extract_entities(
                        chunk,
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        entity_vdb=self.entities_vdb,
                        relationships_vdb=self.relationships_vdb,
                        llm_response_cache=self.llm_response_cache,
                        global_config=asdict(self),
                    )
            else:
                # Standard extraction for original document
                extraction_result = await extract_entities(
                    chunk,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    relationships_vdb=self.relationships_vdb,
                    llm_response_cache=self.llm_response_cache,
                    global_config=asdict(self),
                )
            
            # Check which document this is for tracking
            if hasattr(self, '_last_inserted_doc_id'):
                doc_id = self._last_inserted_doc_id
                
                # Extract entities and relations from the graph for this document
                if doc_id:
                    # Collect entities from this document
                    all_nodes = self.chunk_entity_relation_graph._graph.nodes(data=True)
                    doc_entities = []
                    
                    for node_id, data in all_nodes:
                        if 'source_id' in data:
                            # Check if any source is from the current document's chunks
                            for chunk_key in chunk.keys():
                                if chunk_key in data['source_id'].split(','):
                                    doc_entities.append(node_id)
                                    break
                    
                    # Collect relations from this document
                    all_edges = self.chunk_entity_relation_graph._graph.edges(data=True)
                    doc_relations = []
                    
                    for src, tgt, data in all_edges:
                        if 'source_id' in data:
                            # Check if any source is from the current document's chunks
                            for chunk_key in chunk.keys():
                                if chunk_key in data['source_id'].split(','):
                                    doc_relations.append((src, tgt, data))
                                    break
                    
                    # Store to the appropriate tracking variable
                    if hasattr(self, '_original_entities') and not self._original_entities:
                        self._original_entities = doc_entities
                        self._original_relations = doc_relations
                    elif hasattr(self, '_translated_entities'):
                        self._translated_entities = doc_entities
                        self._translated_relations = doc_relations
            
            # Update the knowledge graph
            if extraction_result is not None:
                self.chunk_entity_relation_graph = extraction_result
                
        except Exception as e:
            logger.error(f"Failed to extract entities and relationships: {e}")
            raise e

    async def _extract_entities_guided(self, chunk: dict[str, Any], original_entities: list[str], original_relations: list) -> BaseGraphStorage:
        """
        Extract entities and relations from translated text with guidance from original entities and relations.
        
        Args:
            chunk: The text chunks to process
            original_entities: List of entities from original document
            original_relations: List of relations from original document
            
        Returns:
            Updated knowledge graph
        """
        # Create a copy of the current knowledge graph
        knowledge_graph = self.chunk_entity_relation_graph
        
        # For each chunk, create a custom prompt to guide entity extraction
        for chunk_key, chunk_data in chunk.items():
            # Get the chunk content
            content = chunk_data.get("content", "")
            
            # Create prompt for guided entity extraction
            prompt = self._create_guided_entity_extraction_prompt(
                content,
                original_entities,
                original_relations
            )
            
            # Get LLM response
            response = await self.llm_model_func(prompt)
            
            # Parse the response to get entities and relations
            entities, relations = self._parse_guided_extraction_response(response)
            
            # Add entities to the knowledge graph
            for entity_name, entity_type, entity_desc in entities:
                # Format entity name
                if not entity_name.startswith('"'):
                    entity_name = f'"{entity_name.upper()}"'
                
                # Create node data
                node_data = {
                    "entity_type": entity_type,
                    "description": entity_desc,
                    "source_id": chunk_key,
                }
                
                # Add to knowledge graph
                await knowledge_graph.upsert_node(entity_name, node_data)
                
                # Also add to vector database
                entity_id = compute_mdhash_id(entity_name, prefix="ent-")
                await self.entities_vdb.upsert({
                    entity_id: {
                        "content": f"{entity_name} {entity_desc}",
                        "entity_name": entity_name,
                    }
                })
            
            # Add relations to the knowledge graph
            for src_entity, tgt_entity, rel_desc, rel_keywords in relations:
                # Format entity names
                if not src_entity.startswith('"'):
                    src_entity = f'"{src_entity.upper()}"'
                if not tgt_entity.startswith('"'):
                    tgt_entity = f'"{tgt_entity.upper()}"'
                
                # Create edge data
                edge_data = {
                    "description": rel_desc,
                    "keywords": rel_keywords,
                    "weight": 1.0,
                    "source_id": chunk_key,
                }
                
                # Ensure both nodes exist
                for entity in [src_entity, tgt_entity]:
                    if not await knowledge_graph.has_node(entity):
                        # Create a placeholder node if it doesn't exist
                        await knowledge_graph.upsert_node(entity, {
                            "entity_type": "UNKNOWN",
                            "description": "Auto-created entity for relation",
                            "source_id": chunk_key,
                        })
                
                # Add edge to knowledge graph
                await knowledge_graph.upsert_edge(src_entity, tgt_entity, edge_data)
                
                # Also add to vector database
                rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
                await self.relationships_vdb.upsert({
                    rel_id: {
                        "content": f"{rel_keywords} {src_entity} {tgt_entity} {rel_desc}",
                        "src_id": src_entity,
                        "tgt_id": tgt_entity,
                    }
                })
        
        return knowledge_graph

    def _create_guided_entity_extraction_prompt(self, content: str, original_entities: list[str], original_relations: list) -> str:
        """
        Create a prompt for guided entity extraction from translated text.
        
        Args:
            content: The text content to extract from
            original_entities: List of entities from original document
            original_relations: List of relations from original document
            
        Returns:
            Prompt for LLM
        """
        # Format original entities
        entities_text = "\n".join([f"- {entity}" for entity in original_entities])
        
        # Format original relations
        relations_text = []
        for src, tgt, data in original_relations:
            relation = f"- {src} → {tgt}: {data.get('description', '')}"
            relations_text.append(relation)
        relations_formatted = "\n".join(relations_text)
        
        prompt = f"""
        You are an expert at identifying named entities and their relationships in text.
        
        I have a translated text, and I need to identify entities and relationships that correspond to those in the original text.
        
        Original entities:
        {entities_text}
        
        Original relationships:
        {relations_formatted}
        
        Translated text:
        {content}
        
        Please identify the corresponding entities and relationships in the translated text.
        
        Format your response as follows:
        
        ENTITIES:
        1. Entity: [entity name] | Type: [entity type] | Description: [brief description]
        2. Entity: [entity name] | Type: [entity type] | Description: [brief description]
        ...
        
        RELATIONSHIPS:
        1. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
        2. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
        ...
        
        It's CRITICAL that you identify entities and relationships that directly correspond to those in the original list.
        Use exactly the same entity types and relationship structures.
        """
        
        return prompt


    def _parse_guided_extraction_response(self, response: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, str]]]:
        """
        Parse the LLM response to extract entities and relations.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (entities, relations)
            - entities: List of (entity_name, entity_type, entity_description)
            - relations: List of (source_entity, target_entity, relation_description, relation_keywords)
        """
        entities = []
        relations = []
        
        # Split into entities and relationships sections
        parts = response.split("RELATIONSHIPS:")
        if len(parts) < 2:
            return entities, relations
        
        entities_section = parts[0].split("ENTITIES:")[1].strip() if "ENTITIES:" in parts[0] else parts[0].strip()
        relationships_section = parts[1].strip()
        
        # Parse entities
        entity_lines = [line.strip() for line in entities_section.split('\n') if line.strip() and '|' in line]
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
                    
                    entities.append((entity_name, entity_type, entity_desc))
            except Exception as e:
                logger.warning(f"Error parsing entity line: {line}. Error: {e}")
        
        # Parse relationships
        relation_lines = [line.strip() for line in relationships_section.split('\n') if line.strip() and '→' in line]
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
                        
                        relations.append((src_entity, tgt_entity, rel_desc, rel_keywords))
            except Exception as e:
                logger.warning(f"Error parsing relationship line: {line}. Error: {e}")
        
        return entities, relations

    async def _translate_preserving_structure(
        self, data: str, target_language: str, 
        source_entities: list[str], source_relations: list
    ) -> str:
        """
        Translate content while preserving entities and relations structure.
        
        Uses LLM to translate the content while maintaining the same entities and relations.
        """
        # Create a prompt for translation that emphasizes preserving entities and relations
        prompt = self._create_translation_prompt(
            data, 
            target_language, 
            source_entities, 
            source_relations
        )
        
        # Get translation from LLM
        response = await self.llm_model_func(prompt)
        
        return response

    def _create_translation_prompt(
        self, data: str, target_language: str, 
        source_entities: list[str], source_relations: list
    ) -> str:
        """
        Create a prompt for the LLM to translate while preserving entities and relations.
        """
        # Format entities for the prompt
        entity_list = "\n".join([f"- {entity}" for entity in source_entities])
        
        # Format relations for the prompt
        relation_list = []
        for src, tgt, data in source_relations:
            relation = f"- {src} → {tgt}: {data.get('description', 'related to')}"
            relation_list.append(relation)
        
        relation_text = "\n".join(relation_list)
        
        prompt = f"""
        You are tasked with translating the following text from its original language to {target_language}.
        
        It is CRITICAL that your translation preserves ALL named entities and relationships between them.
        
        Original Text:
        {data}
        
        Important entities to preserve in translation:
        {entity_list}
        
        Important relationships to preserve in translation:
        {relation_text}
        
        Your translation MUST:
        1. Maintain ALL named entities exactly as they appear
        2. Preserve ALL relationships between entities
        3. Sound natural in {target_language}
        4. Keep the same meaning and tone as the original
        
        Translate the text maintaining all entities and their relationships:
        """
        
        return prompt

    async def _create_cross_lingual_edges(
        self, original_entities: list[str], 
        translated_entities: list[str], target_language: str,
        source_language: str = "Vietnamese"
    ) -> None:
        """
        Create edges in the knowledge graph to connect original entities with their translations.
        
        Args:
            original_entities: List of entities from original text
            translated_entities: List of entities from translated text
            target_language: Target language name
            source_language: Source language name
        """
        # If we have different numbers of entities, we need to align them
        if len(original_entities) != len(translated_entities):
            logger.warning(f"Entity count mismatch: {len(original_entities)} original vs {len(translated_entities)} translated")
            # Use LLM to help align the entities
            aligned_pairs = await self._align_entities(
                original_entities, 
                translated_entities, 
                target_language
            )
        else:
            # Assume entities are in the same order
            aligned_pairs = list(zip(original_entities, translated_entities))
        
        # Create the cross-lingual edges - one edge per pair (undirected graph)
        for orig_entity, trans_entity in aligned_pairs:
            await self.chunk_entity_relation_graph.upsert_edge(
                source_node_id=orig_entity,
                target_node_id=trans_entity,
                edge_data={
                    "description": f"Translation equivalent ({source_language} ↔ {target_language})",
                    "weight": 1.0,
                    "keywords": f"translation,cross-lingual,{source_language},{target_language}",
                    "relation_type": "translation_equivalent",
                    "languages": f"{source_language},{target_language}",
                    "source_id": "cross_lingual",
                    "original_language": source_language,
                    "translated_language": target_language
                }
            )
            
            logger.debug(f"Created cross-lingual edge between {orig_entity} ({source_language}) and {trans_entity} ({target_language})")
    async def _align_entities(
        self, original_entities: list[str], 
        translated_entities: list[str], target_language: str
    ) -> list[tuple[str, str]]:
        """
        Use LLM to align entities from original and translated texts.
        
        Args:
            original_entities: List of entities from original text
            translated_entities: List of entities from translated text
            target_language: Target language name
            
        Returns:
            List of (original_entity, translated_entity) pairs
        """
        # Create prompt for entity alignment
        orig_entities_text = "\n".join([f"{i+1}. {e}" for i, e in enumerate(original_entities)])
        trans_entities_text = "\n".join([f"{i+1}. {e}" for i, e in enumerate(translated_entities)])
        
        prompt = f"""
        I need to align entities between original text and its {target_language} translation.
        
        Original entities:
        {orig_entities_text}
        
        {target_language} entities:
        {trans_entities_text}
        
        For each original entity, find its corresponding translation in the {target_language} list.
        
        Format your response as:
        1. Original: [original entity 1] → Translation: [translated entity 1]
        2. Original: [original entity 2] → Translation: [translated entity 2]
        ...
        
        If an entity doesn't have a corresponding translation, indicate it as "No match".
        """
        
        # Get alignment from LLM
        response = await self.llm_model_func(prompt)
        
        # Parse the response to extract entity pairs
        aligned_pairs = []
        
        # Simple parsing of the response
        lines = response.strip().split('\n')
        for line in lines:
            if '→' in line:
                try:
                    # Extract original and translated entities
                    original_part, translated_part = line.split('→')
                    
                    # Extract entity names from the parts
                    if 'Original:' in original_part and 'Translation:' in translated_part:
                        orig_entity = original_part.split('Original:')[1].strip()
                        trans_entity = translated_part.split('Translation:')[1].strip()
                        
                        # Check if there's a match
                        if trans_entity != "No match" and orig_entity in original_entities and trans_entity in translated_entities:
                            aligned_pairs.append((orig_entity, trans_entity))
                except Exception as e:
                    logger.warning(f"Error parsing alignment line: {line}. Error: {e}")
        
        # Log alignment results
        logger.info(f"Aligned {len(aligned_pairs)} entity pairs between original and {target_language} texts")
        
        return aligned_pairs

    async def _store_translation_pairs(
        self, original_entities: list[str], translated_entities: list[str],
        original_relations: list, translated_relations: list,
        target_language: str, db_path: str
    ) -> None:
        """
        Store entity and relation translation pairs in a JSON file for future use.
        
        Args:
            original_entities: Entities from original text
            translated_entities: Entities from translated text
            original_relations: Relations from original text
            translated_relations: Relations from translated text
            target_language: Target language name
            db_path: Path to save the translation database
        """
        from lightrag.utils import load_json, write_json
        
        # If we have different numbers of entities, we need to align them
        if len(original_entities) != len(translated_entities):
            # Use alignment function to match entities
            entity_pairs = await self._align_entities(
                original_entities, 
                translated_entities, 
                target_language
            )
        else:
            # Assume entities are in the same order
            entity_pairs = list(zip(original_entities, translated_entities))
        
        # Create translation mappings
        entity_translations = []
        for orig, trans in entity_pairs:
            entity_translations.append({
                "original": orig,
                "translated": trans,
                "language": target_language
            })
        
        # For relations, this is more complex as we need to match the relations
        # based on their source and target entities
        relation_translations = []
        
        # Create mapping for quick lookup
        orig_to_trans = {orig: trans for orig, trans in entity_pairs}
        
        # Process relations
        for src, tgt, data in original_relations:
            # Try to find matching relation in translated relations
            if src in orig_to_trans and tgt in orig_to_trans:
                trans_src = orig_to_trans[src]
                trans_tgt = orig_to_trans[tgt]
                
                # Look for matching relation in translated relations
                matching_relation = None
                for t_src, t_tgt, t_data in translated_relations:
                    if t_src == trans_src and t_tgt == trans_tgt:
                        matching_relation = (t_src, t_tgt, t_data)
                        break
                
                if matching_relation:
                    # Add to translation mappings
                    relation_translations.append({
                        "original_src": src,
                        "original_tgt": tgt,
                        "original_desc": data.get('description', ''),
                        "translated_src": trans_src,
                        "translated_tgt": trans_tgt,
                        "translated_desc": matching_relation[2].get('description', ''),
                        "language": target_language
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
                key = f"{trans['original']}|{trans['translated']}|{trans.get('language', '')}"
            elif 'original_src' in trans and 'original_tgt' in trans:
                key = (f"{trans['original_src']}|{trans['original_tgt']}|"
                    f"{trans['translated_src']}|{trans['translated_tgt']}|{trans.get('language', '')}")
            else:
                # If we can't determine a unique key, keep the entry
                unique_translations.append(trans)
                continue
            
            if key not in seen:
                seen.add(key)
                unique_translations.append(trans)
        
        return unique_translations

    async def deduplicate_duo(self, original_doc_id: str, translated_doc_id: str, target_language: str = "Vietnamese"):
        """
        Perform cross-lingual deduplication to ensure both the original and translated
        knowledge graphs maintain the same structure.
        
        Args:
            original_doc_id: ID of the original document
            translated_doc_id: ID of the translated document
            target_language: Target language name
        """
        # Step 1: Get entities from both documents
        original_entities = await self._get_document_entities(original_doc_id)
        translated_entities = await self._get_document_entities(translated_doc_id)
        
        # If we have different numbers of entities, align them
        if len(original_entities) != len(translated_entities):
            # Use alignment function to match entities
            entity_pairs = await self._align_entities(
                original_entities, 
                translated_entities, 
                target_language
            )
        else:
            # Assume entities are in the same order
            entity_pairs = list(zip(original_entities, translated_entities))
        
        # Create mapping dictionaries for quick lookup
        orig_to_trans = {orig: trans for orig, trans in entity_pairs}
        trans_to_orig = {trans: orig for orig, trans in entity_pairs}
        
        # Step 2: Get deduplication mappings in original graph
        # This simulates the normal deduplication process for the original language
        original_dedup_mapping = await self._get_deduplication_mapping(original_entities)
        
        # Step 3: Apply the same deduplication pattern to the translated graph
        translated_dedup_mapping = {}
        for orig_entity, new_orig_entity in original_dedup_mapping.items():
            if orig_entity in orig_to_trans and new_orig_entity in orig_to_trans:
                # Get the corresponding translated entities
                trans_entity = orig_to_trans[orig_entity]
                new_trans_entity = orig_to_trans[new_orig_entity]
                
                # Add to translated mapping
                translated_dedup_mapping[trans_entity] = new_trans_entity
        
        # Step 4: Apply deduplication to original graph
        await self._apply_deduplication(original_dedup_mapping)
        
        # Step 5: Apply deduplication to translated graph
        await self._apply_deduplication(translated_dedup_mapping)
        
        # Step 6: Update cross-lingual links
        await self._update_cross_lingual_links(
            original_dedup_mapping, 
            translated_dedup_mapping, 
            target_language
        )
        
        # Save changes
        await self._insert_done()

    async def _get_document_entities(self, doc_id: str) -> list[str]:
        """
        Get entities associated with a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of entity names
        """
        # Get all chunks for this document
        doc_chunks = {}
        for chunk_id, chunk_data in self.text_chunks._data.items():
            if chunk_data.get('full_doc_id') == doc_id:
                doc_chunks[chunk_id] = chunk_data
        
        if not doc_chunks:
            logger.warning(f"No chunks found for document {doc_id}")
            return []
        
        # Get entities that have these chunks as source
        all_nodes = self.chunk_entity_relation_graph._graph.nodes(data=True)
        doc_entities = []
        
        for node_id, data in all_nodes:
            if 'source_id' in data:
                # Check if any source is from the document's chunks
                sources = data['source_id'].split(',')
                for chunk_id in doc_chunks:
                    if chunk_id in sources:
                        doc_entities.append(node_id)
                        break
        
        return doc_entities

    async def _get_deduplication_mapping(self, entities: list[str]) -> dict[str, str]:
        """
        Simulate the deduplication process and return mapping of original to deduplicated entities.
        
        Args:
            entities: List of entity names
            
        Returns:
            Dictionary mapping original entities to their deduplicated replacements
        """
        # This is a placeholder for the actual deduplication logic
        # In a real implementation, this would implement the same logic used in your
        # main deduplication process
        
        # For example, detecting aliases, similar names, etc.
        dedup_mapping = {}
        
        # Use LLM to identify potential duplicates
        prompt = f"""
        I have a list of entities extracted from a document. I need to identify any duplicates or 
        aliases that refer to the same real-world entity.
        
        Entities:
        {entities}
        
        For each entity that is a duplicate or alias of another entity in the list, provide the mapping.
        Format your response as:
        1. [Entity to merge] should be merged into [Entity to keep]
        2. [Entity to merge] should be merged into [Entity to keep]
        ...
        
        If there are no duplicates, simply respond with "No duplicates found."
        """
        
        response = await self.llm_model_func(prompt)
        
        # Parse the response to extract entity mappings
        if "No duplicates found" not in response:
            lines = response.strip().split('\n')
            for line in lines:
                if "should be merged into" in line:
                    try:
                        # Extract entity names
                        parts = line.split("should be merged into")
                        if len(parts) == 2:
                            entity_to_merge = parts[0].strip().replace("1.", "").replace("2.", "").replace("3.", "").strip()
                            entity_to_keep = parts[1].strip()
                            
                            # Remove any leading numbers and dots
                            entity_to_merge = re.sub(r'^\d+\.\s*', '', entity_to_merge)
                            entity_to_keep = re.sub(r'^\d+\.\s*', '', entity_to_keep)
                            
                            # Add to deduplication mapping
                            if entity_to_merge in entities and entity_to_keep in entities:
                                dedup_mapping[entity_to_merge] = entity_to_keep
                    except Exception as e:
                        logger.warning(f"Error parsing deduplication line: {line}. Error: {e}")
        
        return dedup_mapping

    async def _apply_deduplication(self, dedup_mapping: dict[str, str]) -> None:
        """
        Apply deduplication based on the mapping.
        
        Args:
            dedup_mapping: Dictionary mapping original entities to their deduplicated replacements
        """
        for entity_to_merge, entity_to_keep in dedup_mapping.items():
            # Merge nodes in the graph
            # 1. Get all edges connected to the entity to merge
            if not await self.chunk_entity_relation_graph.has_node(entity_to_merge):
                logger.warning(f"Entity to merge {entity_to_merge} not found in graph")
                continue
                
            if not await self.chunk_entity_relation_graph.has_node(entity_to_keep):
                logger.warning(f"Entity to keep {entity_to_keep} not found in graph")
                continue
            
            # Get original node data
            merge_node_data = await self.chunk_entity_relation_graph.get_node(entity_to_merge)
            keep_node_data = await self.chunk_entity_relation_graph.get_node(entity_to_keep)
            
            if not merge_node_data or not keep_node_data:
                continue
            
            # Merge node data (source_id, description, etc.)
            merged_data = dict(keep_node_data)
            
            # Combine source_ids
            if 'source_id' in merge_node_data and 'source_id' in keep_node_data:
                merged_sources = set(merge_node_data['source_id'].split(','))
                merged_sources.update(keep_node_data['source_id'].split(','))
                merged_data['source_id'] = ','.join(merged_sources)
            
            # Update the node to keep with merged data
            await self.chunk_entity_relation_graph.upsert_node(entity_to_keep, merged_data)
            
            # Get all edges connected to the entity to merge
            connected_edges = await self.chunk_entity_relation_graph.get_node_edges(entity_to_merge)
            if connected_edges:
                # For each edge, create a corresponding edge for the entity to keep
                for src, tgt in connected_edges:
                    edge_data = await self.chunk_entity_relation_graph.get_edge(src, tgt)
                    if edge_data:
                        # If this is an outgoing edge
                        if src == entity_to_merge:
                            # Check if the edge already exists
                            # Check if the edge already exists
                            if not await self.chunk_entity_relation_graph.has_edge(entity_to_keep, tgt):
                                # Create new edge
                                await self.chunk_entity_relation_graph.upsert_edge(
                                    entity_to_keep, tgt, edge_data
                                )
                        # If this is an incoming edge
                        else:
                            # Check if the edge already exists
                            if not await self.chunk_entity_relation_graph.has_edge(src, entity_to_keep):
                                # Create new edge
                                await self.chunk_entity_relation_graph.upsert_edge(
                                    src, entity_to_keep, edge_data
                                )
            
            # Delete the merged entity
            await self.chunk_entity_relation_graph.delete_node(entity_to_merge)
            
            # Update vector databases
            # Delete entity from vector database
            await self.entities_vdb.delete_entity(entity_to_merge)
            
            # Update entity references in relationships vector database
            await self._update_relationship_references(entity_to_merge, entity_to_keep)

    async def _update_relationship_references(self, old_entity: str, new_entity: str) -> None:
        """
        Update references to an entity in the relationships vector database.
        
        Args:
            old_entity: Entity name to be replaced
            new_entity: New entity name
        """
        # Find relationships that reference the old entity
        relations = [
            dp for dp in self.relationships_vdb.client_storage["data"]
            if dp.get("src_id") == old_entity or dp.get("tgt_id") == old_entity
        ]
        
        # For each relationship, create an updated version
        for relation in relations:
            relation_id = relation["__id__"]
            
            # Create updated relation data
            updated_relation = dict(relation)
            
            # Update src_id or tgt_id as needed
            if updated_relation.get("src_id") == old_entity:
                updated_relation["src_id"] = new_entity
            if updated_relation.get("tgt_id") == old_entity:
                updated_relation["tgt_id"] = new_entity
            
            # Update content to reflect the entity change
            if "content" in updated_relation:
                updated_relation["content"] = updated_relation["content"].replace(old_entity, new_entity)
            
            # Delete old relation and insert updated one
            await self.relationships_vdb.delete([relation_id])
            
            # Create a new ID for the updated relation
            new_src = updated_relation.get("src_id", "")
            new_tgt = updated_relation.get("tgt_id", "")
            new_relation_id = compute_mdhash_id(new_src + new_tgt, prefix="rel-")
            
            # Insert the updated relation
            updated_relation["__id__"] = new_relation_id
            self.relationships_vdb._client.upsert([updated_relation])

    async def _update_cross_lingual_links(
        self, 
        original_dedup_mapping: dict[str, str], 
        translated_dedup_mapping: dict[str, str],
        target_language: str
    ) -> None:
        """
        Update cross-lingual links after deduplication.
        
        Args:
            original_dedup_mapping: Mapping of original entities to their deduplicated versions
            translated_dedup_mapping: Mapping of translated entities to their deduplicated versions
            target_language: Target language name
        """
        # Get all edges that represent cross-lingual links
        relation_name = f"translated_to_{target_language.lower()}"
        cross_lingual_edges = []
        
        all_edges = self.chunk_entity_relation_graph._graph.edges(data=True)
        for src, tgt, data in all_edges:
            # Check if this is a cross-lingual edge
            if data.get("source_id") == "cross_lingual":
                cross_lingual_edges.append((src, tgt, data))
        
        # For each cross-lingual edge, check if either endpoint has been deduplicated
        for src, tgt, data in cross_lingual_edges:
            new_src = original_dedup_mapping.get(src, src)
            new_tgt = translated_dedup_mapping.get(tgt, tgt)
            
            # If something changed, update the edge
            if new_src != src or new_tgt != tgt:
                # Delete old edge
                self.chunk_entity_relation_graph._graph.remove_edge(src, tgt)
                
                # Add new edge with same data
                await self.chunk_entity_relation_graph.upsert_edge(
                    new_src, new_tgt, data
                )
                
                logger.debug(f"Updated cross-lingual edge: {src}->{tgt} to {new_src}->{new_tgt}")









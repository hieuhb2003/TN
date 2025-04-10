#!/usr/bin/env python3
"""
Utility script for loading JSON files into vector databases.

This script loads entity, relationship, and chunk data from JSON files
and inserts them into vector databases without requiring a full LightRAG instance.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag import (
    get_openai_embedding_func_with_cache,
    get_openai_embedding_func_with_cache_and_local_embedding_cache,
)
from lightrag.base import (
    BaseVectorStorage,
)
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.operate import load_json_files_to_vector_db
from lightrag.utils import set_logger, EmbeddingFunc, wrap_embedding_func_with_attrs

# Set up logging
set_logger("vector_db_loader.log")

async def load_json_files(
    vector_data_dir: str,
    output_dir: str,
    embedding_model: str = "text-embedding-3-small",
    embedding_dim: int = 1536,
    namespace: str = None,
    manifest_pattern: str = None,
    cosine_threshold: float = 0.7,
):
    """Load JSON files and insert into vector databases
    
    Args:
        vector_data_dir: Directory containing JSON files with vector data
        output_dir: Directory to store vector databases
        embedding_model: Name of the embedding model to use
        embedding_dim: Dimension of the embedding vectors
        namespace: Namespace to filter manifests by
        manifest_pattern: Pattern to match manifest files
        cosine_threshold: Cosine similarity threshold for vector database queries
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create embedding function
    embedding_func = get_openai_embedding_func_with_cache(
        embedding_model, 512, embedding_dim
    )
    
    # Create vector databases
    entities_vdb = NanoVectorDBStorage(
        embedding_func=embedding_func,
        namespace="entities",
        global_config={
            "working_dir": output_dir,
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": cosine_threshold
            }
        }
    )
    
    relationships_vdb = NanoVectorDBStorage(
        embedding_func=embedding_func,
        namespace="relationships",
        global_config={
            "working_dir": output_dir,
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": cosine_threshold
            }
        }
    )
    
    chunks_vdb = NanoVectorDBStorage(
        embedding_func=embedding_func,
        namespace="chunks",
        global_config={
            "working_dir": output_dir,
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": cosine_threshold
            }
        }
    )
    
    # Load JSON files into vector databases
    result = await load_json_files_to_vector_db(
        vector_data_dir=vector_data_dir,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        chunks_vdb=chunks_vdb,
        manifest_pattern=manifest_pattern,
        namespace=namespace,
    )
    
    # Force save all vector databases
    await entities_vdb.force_save()
    await relationships_vdb.force_save()
    await chunks_vdb.force_save()
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Load JSON files into vector databases")
    parser.add_argument(
        "--data-dir", 
        required=True,
        help="Directory containing JSON files with vector data"
    )
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Directory to store vector databases"
    )
    parser.add_argument(
        "--embedding-model", 
        default="text-embedding-3-small",
        help="Name of the embedding model to use"
    )
    parser.add_argument(
        "--embedding-dim", 
        type=int, 
        default=1536,
        help="Dimension of the embedding vectors"
    )
    parser.add_argument(
        "--namespace", 
        help="Namespace to filter manifests by"
    )
    parser.add_argument(
        "--manifest-pattern", 
        help="Pattern to match manifest files"
    )
    parser.add_argument(
        "--cosine-threshold", 
        type=float, 
        default=0.7,
        help="Cosine similarity threshold for vector database queries"
    )
    
    args = parser.parse_args()
    
    # Run the async function
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        load_json_files(
            vector_data_dir=args.data_dir,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            namespace=args.namespace,
            manifest_pattern=args.manifest_pattern,
            cosine_threshold=args.cosine_threshold,
        )
    )
    
    print(f"Loaded {result['entities']} entities, {result['relationships']} relationships, and {result['chunks']} chunks")
    print(f"Vector databases saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Example demonstrating how to use LightRAG with delayed vector database updates.

This example shows:
1. How to insert documents with delay_vector_db_update=True to save to JSON files
2. How to load the JSON files into vector databases later

This approach is useful when:
- You want to separate the graph creation process from vector database creation
- You need to batch-process many documents before creating embeddings
- You're using expensive embedding models and want to minimize API calls
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag import (
    LightRAG,
    get_openai_embedding_func_with_cache,
    get_openai_embedding_func_with_cache_and_local_embedding_cache,
    get_openai_llm_func,
)
from lightrag.utils import set_logger

# Set up logging
set_logger("deferred_vectordb.log")

# Sample documents
sample_documents = [
    """LightRAG is a library for creating and querying knowledge graphs. It uses a combination of 
    LLM-based entity extraction and vector embeddings to create a knowledge graph from documents.""",
    
    """Knowledge graphs are structures that store information about entities and their relationships. 
    They are useful for organizing information and making it accessible through queries.""",
    
    """Vector databases store embeddings of text, which can be used to find semantically similar content.
    In LightRAG, vector databases are used alongside knowledge graphs to provide hybrid search capabilities."""
]

def main():
    # Create working directory if it doesn't exist
    working_dir = os.path.join(os.path.dirname(__file__), "lightrag_working_dir")
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize LightRAG
    print("Initializing LightRAG...")
    lightrag = LightRAG(
        llm_model_func=get_openai_llm_func("gpt-4o"),
        embedding_func=get_openai_embedding_func_with_cache(
            "text-embedding-3-small", 512, 1536
        ),
        enable_llm_cache=True,
        working_dir=working_dir,
    )
    
    # Step 1: Insert documents with delay_vector_db_update=True
    # This will save entity, relationship, and chunk data to JSON files
    # but won't create vector databases yet
    print("\nStep 1: Inserting documents with delayed vector DB updates...")
    lightrag.insert(
        sample_documents,
        language="English",
        delay_vector_db_update=True  # Key parameter to enable JSON file storage
    )
    
    print("\nFiles saved to JSON in: " + os.path.join(working_dir, "vector_data"))
    
    # Step 2: Load the JSON files and create vector databases
    print("\nStep 2: Loading JSON files and creating vector databases...")
    results = lightrag.load_vector_data_from_json()
    
    print(f"\nLoaded {results['entities']} entities, {results['relationships']} relationships, " 
          f"and {results['chunks']} chunks into vector databases")
    
    # Step 3: Test querying - Now we can query as usual
    print("\nStep 3: Testing queries against the loaded vector databases...")
    response = lightrag.query("What is a knowledge graph?")
    
    print("\nQuery response:")
    print(response.answer)
    print("\nReferences:")
    for ref in response.references:
        print(f"- {ref.content}")
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main() 
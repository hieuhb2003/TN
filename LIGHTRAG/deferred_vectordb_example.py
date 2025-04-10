#!/usr/bin/env python
"""
Example script demonstrating how to use deferred vector DB updates in LightRAG

This script shows how to:
1. Insert documents with delayed vector DB updates (only build graph)
2. Save the entity, relationship, and chunk data to JSON files
3. Load the data from JSON files and create vector DBs later

This approach is useful when:
- Processing and creating the graph is faster than embedding
- You want to first build the graph structure and then do heavy embedding operations later
- You want to separate the graph building and vector DB embedding processes
"""

import os
import asyncio
import argparse
from lightrag import LightRAG
from lightrag.utils import always_get_an_event_loop

# Replace with your own embedding and LLM functions
# For example:
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete

async def insert_with_delay(working_dir, documents):
    """
    Insert documents with delayed vector DB updates
    
    Args:
        working_dir: Directory to store graph and JSON files
        documents: List of documents to process
    """
    print(f"Initializing LightRAG in {working_dir}")
    
    # Create working directory if it doesn't exist
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize LightRAG with delay_vector_db_update=True
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    
    print(f"Inserting {len(documents)} documents with delayed vector DB updates")
    
    # Insert documents with delay_vector_db_update=True
    # This will only build the graph and save the data to JSON files
    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}")
        await rag.ainsert(doc, delay_vector_db_update=True)
    
    print("Documents processed and graph built. Vector DB updates were delayed.")
    print(f"JSON files were saved to {working_dir}")
    print("\nTo create vector databases, run:")
    print(f"python load_vector_db.py --working-dir {working_dir}")

async def load_vector_db(working_dir):
    """
    Load vector DB from JSON files
    
    Args:
        working_dir: Directory containing JSON files to load
    """
    print(f"Loading vector DB from JSON files in {working_dir}")
    
    # Initialize LightRAG with the same embedding function
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    
    # Initialize storages
    await rag.initialize_storages()
    
    # Load vector DB from JSON files
    await rag.load_vector_db_from_json()
    
    print("Vector databases created successfully!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demonstrate deferred vector DB updates in LightRAG")
    parser.add_argument("--action", choices=["insert", "load"], required=True, 
                        help="Action to perform: 'insert' to process documents or 'load' to create vector DB")
    parser.add_argument("--working-dir", default="./lightrag_deferred", 
                        help="Working directory for storing data files")
    parser.add_argument("--documents", nargs="+", help="Documents to process (for insert action)")
    
    args = parser.parse_args()
    
    # Get event loop
    loop = always_get_an_event_loop()
    
    if args.action == "insert":
        # Check if documents were provided
        if not args.documents:
            # Use sample documents if none provided
            documents = [
                "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
                "Machine learning is a subset of AI that focuses on training algorithms to make predictions based on data.",
                "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
                "Natural Language Processing (NLP) is a field of AI that enables computers to understand and generate human language."
            ]
        else:
            documents = args.documents
        
        # Insert documents with delayed vector DB updates
        loop.run_until_complete(insert_with_delay(args.working_dir, documents))
    else:
        # Load vector DB from JSON files
        loop.run_until_complete(load_vector_db(args.working_dir))

if __name__ == "__main__":
    main() 
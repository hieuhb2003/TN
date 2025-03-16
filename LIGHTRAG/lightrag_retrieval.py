import os
import json
import asyncio
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.utils import always_get_an_event_loop

def embedding_func_bge_m3(text):
    """
    Create embeddings using BGE-M3 model.
    
    Args:
        text: Text to embed
        
    Returns:
        List of embeddings
    """
    model = SentenceTransformer("BAAI/bge-m3")
    if isinstance(text, list):
        return model.encode(text, normalize_embeddings=True).tolist()
    return model.encode(text, normalize_embeddings=True).tolist()

async def dummy_llm_func(query, system_prompt=None, stream=False):
    """Dummy LLM function that just returns the query."""
    return query

def add_texts_to_lightrag(texts: List[str], working_dir: str = "./lightrag_cache") -> LightRAG:
    """
    Add a list of texts to LightRAG.
    
    Args:
        texts: List of text documents to add
        working_dir: Directory to store LightRAG data
        
    Returns:
        Initialized LightRAG instance
    """
    # Initialize LightRAG
    lightrag = LightRAG(
        working_dir=working_dir,
        embedding_func=embedding_func_bge_m3,
        llm_model_func=dummy_llm_func,
        addon_params={
            "insert_batch_size": 20  # Process 20 documents per batch
        }
    )
    
    # Add texts
    loop = always_get_an_event_loop()
    loop.run_until_complete(_add_texts_async(lightrag, texts))
    
    return lightrag

async def _add_texts_async(lightrag: LightRAG, texts: List[str]) -> None:
    """
    Async implementation of add_texts.
    
    Args:
        lightrag: LightRAG instance
        texts: List of text documents to add
    """
    print(f"Adding {len(texts)} documents to LightRAG...")
    start_time = time.time()
    
    # Use the insert method to add documents
    await lightrag.apipeline_enqueue_documents(texts)
    await lightrag.apipeline_process_enqueue_documents()
    
    elapsed_time = time.time() - start_time
    print(f"Added {len(texts)} documents in {elapsed_time:.2f} seconds")

def retrieve_with_lightrag(
    lightrag: LightRAG,
    query: str,
    mode: str = "hybrid",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Retrieve context for a query using both LightRAG and naive RAG.
    
    Args:
        lightrag: LightRAG instance
        query: The query to retrieve context for
        mode: The retrieval mode to use (hybrid, local, global, naive)
        top_k: Number of top items to retrieve
        
    Returns:
        Dictionary containing the retrieved context and metadata
    """
    loop = always_get_an_event_loop()
    return loop.run_until_complete(_retrieve_async(lightrag, query, mode, top_k))

async def _retrieve_async(
    lightrag: LightRAG,
    query: str,
    mode: str = "hybrid",
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Async implementation of retrieve.
    
    Args:
        lightrag: LightRAG instance
        query: The query to retrieve context for
        mode: The retrieval mode to use (hybrid, local, global, naive)
        top_k: Number of top items to retrieve
        
    Returns:
        Dictionary containing the retrieved context and metadata
    """
    results = {}
    
    # Create query parameters for LightRAG
    lightrag_param = QueryParam(
        mode=mode,
        only_need_context=True,
        top_k=top_k
    )
    
    # Create query parameters for naive RAG
    naive_param = QueryParam(
        mode="naive",
        only_need_context=True,
        top_k=top_k
    )
    
    # Retrieve context using LightRAG
    print(f"Retrieving with LightRAG ({mode} mode)...")
    start_time = time.time()
    lightrag_context = await lightrag.aquery(query, param=lightrag_param)
    lightrag_time = time.time() - start_time
    print(f"LightRAG retrieval time: {lightrag_time:.4f} seconds")
    
    # Retrieve context using naive RAG
    print(f"Retrieving with naive RAG...")
    start_time = time.time()
    naive_context = await lightrag.aquery(query, param=naive_param)
    naive_time = time.time() - start_time
    print(f"Naive RAG retrieval time: {naive_time:.4f} seconds")
    
    # Parse the contexts to extract the sources
    lightrag_sources = parse_context_sources(lightrag_context)
    naive_sources = parse_context_sources(naive_context)
    
    # Get full documents for the chunks
    lightrag_full_docs = await get_full_documents(lightrag, lightrag_sources)
    naive_full_docs = await get_full_documents(lightrag, naive_sources)
    
    results = {
        "query": query,
        "lightrag": {
            "mode": mode,
            "context": lightrag_context,
            "sources": lightrag_sources,
            "full_docs": lightrag_full_docs,
            "time": lightrag_time
        },
        "naive": {
            "context": naive_context,
            "sources": naive_sources,
            "full_docs": naive_full_docs,
            "time": naive_time
        }
    }
    
    return results

def parse_context_sources(context: str) -> List[Dict[str, str]]:
    """
    Parse the context string to extract the sources.
    
    Args:
        context: The context string returned by LightRAG
        
    Returns:
        List of dictionaries containing source information
    """
    if not context:
        return []
    
    sources = []
    sections = context.split("-----")
    
    for section in sections:
        if "Sources" in section:
            lines = section.strip().split("\n")
            # Skip the header and the CSV header
            content_lines = [line for line in lines if not line.startswith("```") and "id,content" not in line]
            
            for line in content_lines:
                if "," in line:
                    # Split at the first comma to separate id and content
                    chunk_id, content = line.split(",", 1)
                    sources.append({
                        "id": chunk_id.strip(),
                        "content": content.strip()
                    })
    
    return sources

async def get_full_documents(lightrag: LightRAG, sources: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get the full documents for a list of chunk sources.
    
    Args:
        lightrag: LightRAG instance
        sources: List of dictionaries containing source information
        
    Returns:
        Dictionary mapping document IDs to full document content
    """
    full_docs = {}
    
    for source in sources:
        chunk_id = source["id"]
        # Get the chunk data which contains the full_doc_id
        chunk_data = await lightrag.text_chunks.get_by_id(chunk_id)
        
        if chunk_data:
            full_doc_id = chunk_data.get("full_doc_id")
            if full_doc_id and full_doc_id not in full_docs:
                # Get the full document content
                full_doc = await lightrag.full_docs.get_by_id(full_doc_id)
                if full_doc:
                    full_docs[full_doc_id] = full_doc["content"]
    
    return full_docs

def print_retrieval_results(results: Dict[str, Any]) -> None:
    """
    Print the retrieval results in a readable format.
    
    Args:
        results: Dictionary containing retrieval results
    """
    print("\n" + "="*80)
    print(f"QUERY: {results['query']}")
    print("="*80)
    
    # Print LightRAG results
    print(f"\nLIGHTRAG RESULTS ({results['lightrag']['mode']} mode):")
    print(f"Retrieval time: {results['lightrag']['time']:.4f} seconds")
    print("\nRetrieved chunks:")
    for i, source in enumerate(results['lightrag']['sources']):
        print(f"\n{i+1}. Chunk ID: {source['id']}")
        print(f"   Content: {source['content'][:100]}...")
    
    print("\nFull documents:")
    for i, (doc_id, content) in enumerate(results['lightrag']['full_docs'].items()):
        print(f"\n{i+1}. Document ID: {doc_id}")
        print(f"   Content: {content[:100]}...")
    
    # Print naive RAG results
    print("\n" + "-"*80)
    print("\nNAIVE RAG RESULTS:")
    print(f"Retrieval time: {results['naive']['time']:.4f} seconds")
    print("\nRetrieved chunks:")
    for i, source in enumerate(results['naive']['sources']):
        print(f"\n{i+1}. Chunk ID: {source['id']}")
        print(f"   Content: {source['content'][:100]}...")
    
    print("\nFull documents:")
    for i, (doc_id, content) in enumerate(results['naive']['full_docs'].items()):
        print(f"\n{i+1}. Document ID: {doc_id}")
        print(f"   Content: {content[:100]}...")
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON:")
    print(f"LightRAG ({results['lightrag']['mode']} mode): {results['lightrag']['time']:.4f} seconds")
    print(f"Naive RAG: {results['naive']['time']:.4f} seconds")
    
    if results['lightrag']['time'] > 0:
        speedup = results['naive']['time'] / results['lightrag']['time']
        print(f"Speed improvement: {speedup:.2f}x")
    
    lightrag_doc_ids = set(results['lightrag']['full_docs'].keys())
    naive_doc_ids = set(results['naive']['full_docs'].keys())
    
    common_docs = lightrag_doc_ids.intersection(naive_doc_ids)
    lightrag_unique = lightrag_doc_ids - naive_doc_ids
    naive_unique = naive_doc_ids - lightrag_doc_ids
    
    print(f"\nCommon documents: {len(common_docs)}")
    print(f"Documents unique to LightRAG: {len(lightrag_unique)}")
    print(f"Documents unique to naive RAG: {len(naive_unique)}")
    print("="*80)

# Example usage
if __name__ == "__main__":
    # Example texts to add
    texts = [
        "LightRAG is a simple and fast Retrieval-Augmented Generation system.",
        "BGE-M3 is a powerful embedding model for text retrieval.",
        "Retrieval-Augmented Generation combines retrieval and generation for better results.",
        "Knowledge graphs can improve retrieval by capturing relationships between entities.",
        "Vector databases store embeddings for efficient similarity search."
    ]
    
    # Add texts to LightRAG
    lightrag = add_texts_to_lightrag(texts)
    
    # Example query
    query = "What is LightRAG and how does it work?"
    
    # Retrieve with LightRAG and naive RAG
    results = retrieve_with_lightrag(lightrag, query, mode="hybrid", top_k=3)
    
    # Print results
    print_retrieval_results(results)
    
    # Save results to file
    with open("retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to retrieval_results.json") 
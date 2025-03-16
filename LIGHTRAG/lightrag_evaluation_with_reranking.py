import os
import json
import asyncio
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.util import cos_sim

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

async def retrieve_documents(
    lightrag: LightRAG,
    query: str,
    mode: str = "hybrid",
    top_k: int = 10
) -> Tuple[List[str], List[str]]:
    """
    Retrieve documents for a query using LightRAG.
    
    Args:
        lightrag: LightRAG instance
        query: The query to retrieve documents for
        mode: The retrieval mode to use (hybrid, local, global, naive)
        top_k: Number of top items to retrieve
        
    Returns:
        Tuple of (lightrag_docs, naive_docs) where each is a list of document contents
    """
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
    lightrag_context = await lightrag.aquery(query, param=lightrag_param)
    
    # Retrieve context using naive RAG
    naive_context = await lightrag.aquery(query, param=naive_param)
    
    # Parse the contexts to extract the sources
    lightrag_sources = parse_context_sources(lightrag_context)
    naive_sources = parse_context_sources(naive_context)
    
    # Get full documents for the chunks
    lightrag_full_docs = await get_full_documents(lightrag, lightrag_sources)
    naive_full_docs = await get_full_documents(lightrag, naive_sources)
    
    # Extract document contents
    lightrag_docs = list(lightrag_full_docs.values())
    naive_docs = list(naive_full_docs.values())
    
    return lightrag_docs, naive_docs

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

def create_reranking_samples(
    queries: List[str],
    lightrag_docs_list: List[List[str]],
    naive_docs_list: List[List[str]],
    ground_truth_docs: List[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Create samples for RerankingEvaluator.
    
    Args:
        queries: List of queries
        lightrag_docs_list: List of lists of documents retrieved by LightRAG for each query
        naive_docs_list: List of lists of documents retrieved by naive RAG for each query
        ground_truth_docs: Optional list of lists of ground truth relevant documents for each query
        
    Returns:
        List of samples for RerankingEvaluator
    """
    samples = []
    
    for i, query in enumerate(queries):
        lightrag_docs = lightrag_docs_list[i]
        naive_docs = naive_docs_list[i]
        
        # If ground truth is provided, use it
        if ground_truth_docs is not None:
            positive_docs = ground_truth_docs[i]
            # All retrieved docs that are not in ground truth are considered negative
            negative_docs = []
            for doc in lightrag_docs + naive_docs:
                if doc not in positive_docs and doc not in negative_docs:
                    negative_docs.append(doc)
        else:
            # If no ground truth, assume LightRAG docs are positive and naive docs are negative
            # This is just for comparison purposes
            positive_docs = lightrag_docs
            negative_docs = []
            for doc in naive_docs:
                if doc not in positive_docs and doc not in negative_docs:
                    negative_docs.append(doc)
        
        # Create sample
        sample = {
            "query": query,
            "positive": positive_docs,
            "negative": negative_docs
        }
        
        samples.append(sample)
    
    return samples

def evaluate_retrieval_with_reranking(
    lightrag: LightRAG,
    queries: List[str],
    ground_truth_docs: List[List[str]] = None,
    mode: str = "hybrid",
    top_k: int = 10,
    model_name: str = "BAAI/bge-m3",
    output_path: str = None
) -> Dict[str, float]:
    """
    Evaluate retrieval performance using RerankingEvaluator.
    
    Args:
        lightrag: LightRAG instance
        queries: List of queries to evaluate
        ground_truth_docs: Optional list of lists of ground truth relevant documents for each query
        mode: The retrieval mode to use for LightRAG
        top_k: Number of top items to retrieve
        model_name: Name of the model to use for evaluation
        output_path: Path to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Retrieve documents for each query
    lightrag_docs_list = []
    naive_docs_list = []
    
    loop = always_get_an_event_loop()
    
    print(f"Retrieving documents for {len(queries)} queries...")
    for query in queries:
        lightrag_docs, naive_docs = loop.run_until_complete(
            retrieve_documents(lightrag, query, mode, top_k)
        )
        lightrag_docs_list.append(lightrag_docs)
        naive_docs_list.append(naive_docs)
    
    # Create samples for RerankingEvaluator
    samples = create_reranking_samples(queries, lightrag_docs_list, naive_docs_list, ground_truth_docs)
    
    # Initialize SentenceTransformer model
    model = SentenceTransformer(model_name)
    
    # Initialize RerankingEvaluator
    evaluator = RerankingEvaluator(
        samples=samples,
        at_k=top_k,
        name=f"LightRAG_{mode}_vs_Naive",
        write_csv=True,
        show_progress_bar=True
    )
    
    # Evaluate
    print(f"Evaluating retrieval performance using {model_name}...")
    metrics = evaluator(model, output_path=output_path)
    
    return metrics

def evaluate_lightrag_vs_naive(
    lightrag: LightRAG,
    queries: List[str],
    ground_truth_docs: List[List[str]] = None,
    modes: List[str] = ["hybrid", "local", "global", "naive"],
    top_k: int = 10,
    model_name: str = "BAAI/bge-m3",
    output_path: str = "./evaluation_results"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate LightRAG vs naive RAG for different modes.
    
    Args:
        lightrag: LightRAG instance
        queries: List of queries to evaluate
        ground_truth_docs: Optional list of lists of ground truth relevant documents for each query
        modes: List of modes to evaluate
        top_k: Number of top items to retrieve
        model_name: Name of the model to use for evaluation
        output_path: Path to save evaluation results
        
    Returns:
        Dictionary mapping modes to evaluation metrics
    """
    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    results = {}
    
    for mode in modes:
        print(f"\nEvaluating mode: {mode}")
        metrics = evaluate_retrieval_with_reranking(
            lightrag=lightrag,
            queries=queries,
            ground_truth_docs=ground_truth_docs,
            mode=mode,
            top_k=top_k,
            model_name=model_name,
            output_path=output_path
        )
        
        results[mode] = metrics
        
        print(f"Mode: {mode}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save results to file
    if output_path:
        with open(os.path.join(output_path, "evaluation_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    return results

# Example usage
if __name__ == "__main__":
    # Example texts to add
    texts = [
        "LightRAG is a simple and fast Retrieval-Augmented Generation system.",
        "BGE-M3 is a powerful embedding model for text retrieval.",
        "Retrieval-Augmented Generation combines retrieval and generation for better results.",
        "Knowledge graphs can improve retrieval by capturing relationships between entities.",
        "Vector databases store embeddings for efficient similarity search.",
        "LightRAG uses a hybrid approach combining knowledge graphs and vector search.",
        "BGE-M3 is based on the BERT architecture with improvements for multilingual support.",
        "RAG systems retrieve relevant documents and then generate responses based on them.",
        "Knowledge graphs represent relationships between entities as a graph structure.",
        "Vector databases use approximate nearest neighbor search for efficient retrieval."
    ]
    
    # Add texts to LightRAG
    lightrag = add_texts_to_lightrag(texts)
    
    # Example queries
    queries = [
        "What is LightRAG?",
        "How does BGE-M3 work?",
        "What are the benefits of knowledge graphs?",
        "How does vector search work?"
    ]
    
    # Example ground truth (optional)
    ground_truth = [
        ["LightRAG is a simple and fast Retrieval-Augmented Generation system.", 
         "LightRAG uses a hybrid approach combining knowledge graphs and vector search."],
        ["BGE-M3 is a powerful embedding model for text retrieval.",
         "BGE-M3 is based on the BERT architecture with improvements for multilingual support."],
        ["Knowledge graphs can improve retrieval by capturing relationships between entities.",
         "Knowledge graphs represent relationships between entities as a graph structure."],
        ["Vector databases store embeddings for efficient similarity search.",
         "Vector databases use approximate nearest neighbor search for efficient retrieval."]
    ]
    
    # Evaluate LightRAG vs naive RAG
    results = evaluate_lightrag_vs_naive(
        lightrag=lightrag,
        queries=queries,
        ground_truth_docs=ground_truth,
        modes=["hybrid", "local", "global", "naive"],
        top_k=5,
        model_name="BAAI/bge-m3",
        output_path="./evaluation_results"
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    for mode, metrics in results.items():
        print(f"\nMode: {mode}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}") 
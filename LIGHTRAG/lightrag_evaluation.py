import os
import json
import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict
import numpy as np
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.namespace import NameSpace, make_namespace
from lightrag.utils import compute_mdhash_id, always_get_an_event_loop

class LightRAGEvaluator:
    """
    A class to evaluate retrieval performance between LightRAG and naive RAG using BGE-M3.
    """
    
    def __init__(
        self, 
        working_dir: str = "./lightrag_cache",
        embedding_model_name: str = "BAAI/bge-m3",
        use_existing_data: bool = True
    ):
        """
        Initialize the LightRAG evaluator.
        
        Args:
            working_dir: Directory to store LightRAG data
            embedding_model_name: Name of the embedding model to use
            use_existing_data: Whether to use existing data in the working_dir
        """
        self.working_dir = working_dir
        self.embedding_model_name = embedding_model_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Create embedding function
        def embedding_func(text):
            if isinstance(text, list):
                return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
            return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
        
        # Initialize LightRAG
        self.lightrag = LightRAG(
            working_dir=working_dir,
            embedding_func=embedding_func,
            llm_model_func=self._dummy_llm_func,  # We don't need LLM for evaluation
            addon_params={
                "insert_batch_size": 20  # Process 20 documents per batch
            }
        )
        
        # Load existing data if requested
        if use_existing_data:
            self._load_existing_data()
    
    async def _dummy_llm_func(self, query, system_prompt=None, stream=False):
        """Dummy LLM function that just returns the query."""
        return query
    
    def _load_existing_data(self):
        """Load existing data from kv_store_text_chunks.json and kv_store_full_docs.json."""
        try:
            # Check if the files exist
            chunks_path = os.path.join(self.working_dir, "kv_store_text_chunks.json")
            docs_path = os.path.join(self.working_dir, "kv_store_full_docs.json")
            
            if os.path.exists(chunks_path) and os.path.exists(docs_path):
                print(f"Found existing data in {self.working_dir}")
                print(f"Chunks: {chunks_path}")
                print(f"Full docs: {docs_path}")
            else:
                print(f"No existing data found in {self.working_dir}")
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
    def add_texts(self, texts: List[str]) -> None:
        """
        Add a list of texts to LightRAG.
        
        Args:
            texts: List of text documents to add
        """
        loop = always_get_an_event_loop()
        loop.run_until_complete(self._add_texts_async(texts))
    
    async def _add_texts_async(self, texts: List[str]) -> None:
        """
        Async implementation of add_texts.
        
        Args:
            texts: List of text documents to add
        """
        print(f"Adding {len(texts)} documents to LightRAG...")
        start_time = time.time()
        
        # Use the insert method to add documents
        await self.lightrag.apipeline_enqueue_documents(texts)
        await self.lightrag.apipeline_process_enqueue_documents()
        
        elapsed_time = time.time() - start_time
        print(f"Added {len(texts)} documents in {elapsed_time:.2f} seconds")
    
    def retrieve(
        self, 
        query: str, 
        mode: str = "hybrid", 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve context for a query using both LightRAG and naive RAG.
        
        Args:
            query: The query to retrieve context for
            mode: The retrieval mode to use (hybrid, local, global, naive)
            top_k: Number of top items to retrieve
            
        Returns:
            Dictionary containing the retrieved context and metadata
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self._retrieve_async(query, mode, top_k))
    
    async def _retrieve_async(
        self, 
        query: str, 
        mode: str = "hybrid", 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Async implementation of retrieve.
        
        Args:
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
        start_time = time.time()
        lightrag_context = await self.lightrag.aquery(query, param=lightrag_param)
        lightrag_time = time.time() - start_time
        
        # Retrieve context using naive RAG
        start_time = time.time()
        naive_context = await self.lightrag.aquery(query, param=naive_param)
        naive_time = time.time() - start_time
        
        # Parse the contexts to extract the sources
        lightrag_sources = self._parse_context_sources(lightrag_context)
        naive_sources = self._parse_context_sources(naive_context)
        
        # Get full documents for the chunks
        lightrag_full_docs = await self._get_full_documents(lightrag_sources)
        naive_full_docs = await self._get_full_documents(naive_sources)
        
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
    
    def _parse_context_sources(self, context: str) -> List[Dict[str, str]]:
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
    
    async def _get_full_documents(self, sources: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Get the full documents for a list of chunk sources.
        
        Args:
            sources: List of dictionaries containing source information
            
        Returns:
            Dictionary mapping document IDs to full document content
        """
        full_docs = {}
        
        for source in sources:
            chunk_id = source["id"]
            # Get the chunk data which contains the full_doc_id
            chunk_data = await self.lightrag.text_chunks.get_by_id(chunk_id)
            
            if chunk_data:
                full_doc_id = chunk_data.get("full_doc_id")
                if full_doc_id and full_doc_id not in full_docs:
                    # Get the full document content
                    full_doc = await self.lightrag.full_docs.get_by_id(full_doc_id)
                    if full_doc:
                        full_docs[full_doc_id] = full_doc["content"]
        
        return full_docs
    
    def evaluate(
        self, 
        queries: List[str], 
        mode: str = "hybrid", 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance for a list of queries.
        
        Args:
            queries: List of queries to evaluate
            mode: The retrieval mode to use for LightRAG
            top_k: Number of top items to retrieve
            
        Returns:
            Dictionary containing evaluation metrics
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self._evaluate_async(queries, mode, top_k))
    
    async def _evaluate_async(
        self, 
        queries: List[str], 
        mode: str = "hybrid", 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Async implementation of evaluate.
        
        Args:
            queries: List of queries to evaluate
            mode: The retrieval mode to use for LightRAG
            top_k: Number of top items to retrieve
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = []
        lightrag_times = []
        naive_times = []
        
        for query in queries:
            print(f"Evaluating query: {query}")
            result = await self._retrieve_async(query, mode, top_k)
            results.append(result)
            
            lightrag_times.append(result["lightrag"]["time"])
            naive_times.append(result["naive"]["time"])
        
        # Calculate average retrieval times
        avg_lightrag_time = sum(lightrag_times) / len(lightrag_times)
        avg_naive_time = sum(naive_times) / len(naive_times)
        
        evaluation = {
            "queries": len(queries),
            "mode": mode,
            "top_k": top_k,
            "avg_lightrag_time": avg_lightrag_time,
            "avg_naive_time": avg_naive_time,
            "results": results
        }
        
        return evaluation
    
    def save_evaluation(self, evaluation: Dict[str, Any], filename: str = "evaluation_results.json") -> None:
        """
        Save evaluation results to a file.
        
        Args:
            evaluation: Evaluation results to save
            filename: Name of the file to save to
        """
        with open(filename, "w") as f:
            json.dump(evaluation, f, indent=2)
        
        print(f"Saved evaluation results to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize the evaluator
    evaluator = LightRAGEvaluator(
        working_dir="./lightrag_cache",
        embedding_model_name="BAAI/bge-m3"
    )
    
    # Example texts to add
    texts = [
        "LightRAG is a simple and fast Retrieval-Augmented Generation system.",
        "BGE-M3 is a powerful embedding model for text retrieval.",
        "Retrieval-Augmented Generation combines retrieval and generation for better results.",
        "Knowledge graphs can improve retrieval by capturing relationships between entities.",
        "Vector databases store embeddings for efficient similarity search."
    ]
    
    # Add texts to LightRAG
    evaluator.add_texts(texts)
    
    # Example queries to evaluate
    queries = [
        "What is LightRAG?",
        "How does BGE-M3 work?",
        "What are the benefits of knowledge graphs?",
        "How does vector search work?"
    ]
    
    # Evaluate retrieval performance
    evaluation = evaluator.evaluate(queries, mode="hybrid", top_k=3)
    
    # Save evaluation results
    evaluator.save_evaluation(evaluation)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Queries: {evaluation['queries']}")
    print(f"Mode: {evaluation['mode']}")
    print(f"Top-K: {evaluation['top_k']}")
    print(f"Avg LightRAG Time: {evaluation['avg_lightrag_time']:.4f} seconds")
    print(f"Avg Naive RAG Time: {evaluation['avg_naive_time']:.4f} seconds")
    print(f"Speed Improvement: {evaluation['avg_naive_time'] / evaluation['avg_lightrag_time']:.2f}x") 
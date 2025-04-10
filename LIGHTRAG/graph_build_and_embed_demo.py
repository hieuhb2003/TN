"""
LightRAG Graph Building and Deferred Embedding Demo

This script demonstrates how to:
1. Build a knowledge graph without embedding to vector database (faster)
2. Save graph data to text files
3. Load and embed graph data into vector databases later

This approach is useful when:
- Inserting and saving chunks/entities/relations to embedding takes a lot of time
- You want to build the graph first and defer the vector DB operations
- You have multiple embedding models you want to use with the same graph
"""

import asyncio
import os
import time
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop

# Sample document for demonstration
SAMPLE_DOCUMENT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, 
as opposed to the natural intelligence displayed by animals and humans. 
AI research has been defined as the field of study of intelligent agents, 
which refers to any system that perceives its environment and takes actions 
that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe 
machines that mimic and display "human" cognitive skills that are associated 
with the human mind, such as "learning" and "problem-solving". This definition 
has since been rejected by major AI researchers who now describe AI in terms 
of rationality and acting rationally, which does not limit how intelligence 
can be articulated.

AI applications include advanced web search engines (e.g., Google), recommendation 
systems (used by YouTube, Amazon, and Netflix), understanding human speech 
(such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or 
creative tools (ChatGPT and AI art), automated decision-making, and competing 
at the highest level in strategic game systems (such as chess and Go).
"""

# Simple mock embedding function for demonstration
async def mock_embedding_func(text):
    """A simple mock embedding function that returns random vectors"""
    import numpy as np
    # Simulate some computation time
    await asyncio.sleep(0.05)
    # Create a deterministic embedding based on the content hash
    if isinstance(text, list):
        return [np.random.rand(1536).tolist() for _ in text]
    return np.random.rand(1536).tolist()

# Simple mock LLM function for demonstration
async def mock_llm_func(prompt, **kwargs):
    """A simple mock LLM function"""
    await asyncio.sleep(0.1)  # Simulate LLM processing time
    
    if "entity" in prompt.lower() and "extract" in prompt.lower():
        return """(PERSON, "AI Researcher", "A scientist specializing in AI research")
(CONCEPT, "Artificial Intelligence", "Intelligence demonstrated by machines")
(CONCEPT, "Natural Intelligence", "Intelligence displayed by animals and humans")
(ORGANIZATION, "Google", "A technology company that develops search engines")
(PRODUCT, "Siri", "Apple's virtual assistant")
(PRODUCT, "Alexa", "Amazon's virtual assistant")
(PRODUCT, "ChatGPT", "An AI chatbot developed by OpenAI")"""
    
    if "relation" in prompt.lower():
        return """(CONCEPT:"Artificial Intelligence" -> CONCEPT:"Natural Intelligence", "is contrasted with", "comparison,difference")
(ORGANIZATION:"Google" -> PRODUCT:"Web search engines", "develops", "technology,creation")
(PRODUCT:"Siri" -> CONCEPT:"Understanding human speech", "enables", "capability,feature")"""
    
    return "Generated response for: " + prompt[:50] + "..."

# Alternative embedding function to demonstrate using different embeddings
async def alternative_embedding_func(text):
    """A different mock embedding function"""
    import numpy as np
    await asyncio.sleep(0.08)  # Slightly slower
    # Use different random seed to get different embeddings
    if isinstance(text, list):
        return [np.random.rand(1536).tolist() for _ in text]
    return (np.random.rand(1536) * 2).tolist()  # Different scale

async def main():
    """Main function demonstrating the workflow"""
    working_dir = "./graph_build_demo"
    os.makedirs(working_dir, exist_ok=True)
    
    # Method 1: Using enable_vdb_upsert parameter in LightRAG constructor
    print("\n1. Creating LightRAG instance with vector database upsert disabled")
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=mock_embedding_func,
        llm_model_func=mock_llm_func,
        enable_vdb_upsert=False  # Disable vector DB operations during construction
    )
    
    # Initialize storages
    print("2. Initializing storages")
    await rag.initialize_storages()
    
    # Process the document to build the graph (without embedding to vector DB)
    print("3. Building the graph (without vector DB embeddings)")
    start_time = time.time()
    await rag.apipeline_enqueue_documents(SAMPLE_DOCUMENT)
    await rag.apipeline_process_enqueue_documents()
    graph_build_time = time.time() - start_time
    print(f"   Graph built in {graph_build_time:.2f} seconds")
    
    # Save the graph data to files
    print("4. Saving graph data to files")
    output_dir = await rag.save_graph_data()
    print(f"   Graph data saved to {output_dir}")
    
    # Method 2: Using enable_embedding parameter in insert method
    print("\n5. Creating another LightRAG instance with default settings")
    rag2 = LightRAG(
        working_dir=working_dir + "_method2",
        embedding_func=mock_embedding_func,
        llm_model_func=mock_llm_func,
        # Default is enable_vdb_upsert=True
    )
    
    # Process the document with embedding disabled for this operation only
    print("6. Using insert with enable_embedding=False parameter")
    await rag2.initialize_storages()
    start_time = time.time()
    # Use the enable_embedding parameter to temporarily disable embedding
    rag2.insert(SAMPLE_DOCUMENT, enable_embedding=False)
    graph_build_time = time.time() - start_time
    print(f"   Graph built in {graph_build_time:.2f} seconds without embedding")
    
    # Save the graph data
    print("7. Saving graph data from second method")
    output_dir2 = await rag2.save_graph_data()
    print(f"   Graph data saved to {output_dir2}")
    
    # Now load and embed the graph data with the first embedding function
    print("\n8. Loading and embedding the graph data with first embedding function")
    start_time = time.time()
    await rag.load_and_embed_graph(
        input_dir=output_dir,
        embedding_func=mock_embedding_func,
        embedding_name="default_embedding"
    )
    vdb_embed_time = time.time() - start_time
    print(f"   Vector databases embedded in {vdb_embed_time:.2f} seconds")
    
    # Load and embed with alternative embedding function
    print("\n9. Re-embedding with a different embedding function")
    start_time = time.time()
    await rag.load_and_embed_graph(
        input_dir=output_dir,
        embedding_func=alternative_embedding_func,
        embedding_name="alternative_embedding"
    )
    vdb_reembed_time = time.time() - start_time
    print(f"   Alternative vector databases embedded in {vdb_reembed_time:.2f} seconds")
    
    # List created vector database files
    vdb_files = [f for f in os.listdir(working_dir) if f.startswith('vdb_')]
    print("\n10. Vector database files created:")
    for f in vdb_files:
        print(f"   - {f}")
    
    # Finalize storages
    print("\n11. Finalizing storages")
    await rag.finalize_storages()
    await rag2.finalize_storages()
    
    print("\nComplete! The graph was built once and embedded twice with different functions.")

def run_demo():
    """Run the main async function"""
    loop = always_get_an_event_loop()
    loop.run_until_complete(main())

if __name__ == "__main__":
    run_demo() 
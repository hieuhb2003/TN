"""
Example script demonstrating how to decouple graph building from vector database embedding in LightRAG.

This approach allows you to:
1. Build and serialize a knowledge graph (without embedding)
2. Later load and embed the graph into vector databases with different embedding models

This is useful when:
- You want to reuse the same graph with different embedding models
- The embedding process is computationally expensive and you want to separate it
- You're building a large graph that takes time and you want to ensure it's saved before embedding
"""

import asyncio
import os
import sys
from typing import List, Callable, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.lightrag import LightRAG
from lightrag.utils import limit_async_func_call

# Example document
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

As machines become increasingly capable, tasks considered to require "intelligence" 
are often removed from the definition of AI, a phenomenon known as the AI effect. 
For instance, optical character recognition is frequently excluded from things 
considered to be AI, having become a routine technology.
"""

class EmbeddingModels:
    """Simple class with different embedding functions for demonstration"""
    
    @staticmethod
    async def random_embedding(text: str) -> List[float]:
        """A dummy embedding function that returns random vectors"""
        import numpy as np
        # Simulate some processing time
        await asyncio.sleep(0.01)
        return np.random.rand(1536).astype(np.float32)
    
    @staticmethod
    async def mock_bge_embedding(text: str) -> List[float]:
        """Mock BGE embedding function"""
        import numpy as np
        # Simulate some processing time
        await asyncio.sleep(0.05)
        # Use the hash of the text to create a deterministic but unique embedding
        hash_val = hash(text)
        np.random.seed(hash_val)
        return np.random.rand(1536).astype(np.float32)
    
    @staticmethod
    async def mock_openai_embedding(text: str) -> List[float]:
        """Mock OpenAI embedding function"""
        import numpy as np
        # Simulate some processing time
        await asyncio.sleep(0.1)
        # Use the hash of the text to create a deterministic but unique embedding
        hash_val = hash(text)
        np.random.seed(hash_val + 42)  # Different seed than BGE
        return np.random.rand(1536).astype(np.float32)


async def dummy_llm_model_func(prompt: str, **kwargs) -> str:
    """A dummy LLM function for demonstration purposes"""
    # Return some content based on the prompt
    if "entity_extraction" in prompt.lower():
        return """(PERSON, "AI Researcher", "A scientist specializing in artificial intelligence")
(CONCEPT, "Artificial Intelligence", "Intelligence demonstrated by machines")
(ORGANIZATION, "Google", "A technology company that develops advanced web search engines")
(ORGANIZATION, "Amazon", "An e-commerce and cloud computing company")
(ORGANIZATION, "Netflix", "A streaming service company")
(PRODUCT, "Siri", "Apple's voice assistant")
(PRODUCT, "Alexa", "Amazon's voice assistant")
(ORGANIZATION, "Waymo", "A self-driving technology company")
(PRODUCT, "ChatGPT", "An AI chatbot developed by OpenAI")
(CONCEPT, "AI Effect", "The phenomenon where AI achievements are no longer considered AI")"""
    elif "continue_extraction" in prompt.lower():
        return """(CONCEPT, "Natural Intelligence", "Intelligence displayed by animals and humans")
(CONCEPT, "Intelligent Agents", "Systems that perceive their environment and take actions")
(ACTIVITY, "Learning", "The acquisition of knowledge or skills through experience")
(ACTIVITY, "Problem-Solving", "The process of finding solutions to difficult issues")
(PRODUCT, "Self-driving cars", "Vehicles capable of sensing their environment and operating without human involvement")"""
    elif "relationship_extraction" in prompt.lower():
        return """(CONCEPT:"Artificial Intelligence" -> CONCEPT:"Intelligent Agents", "AI is defined as the study of", "definition,research,study")
(CONCEPT:"Artificial Intelligence" -> CONCEPT:"Natural Intelligence", "is contrasted with", "comparison,difference,opposite")
(ORGANIZATION:"Google" -> PRODUCT:"Web search engines", "develops", "creation,development,technology")
(ORGANIZATION:"Amazon" -> PRODUCT:"Alexa", "develops", "creation,product,voice assistant")
(CONCEPT:"AI Effect" -> CONCEPT:"Artificial Intelligence", "redefines", "definition,evolution,perception")"""
    
    # Generic response
    return "This is a simulated response from the LLM model."


async def main():
    """Main function demonstrating the decoupled graph building and embedding process"""
    # Create working directory
    working_dir = "./test_decoupled"
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize embedding functions with rate limiting
    random_embed_func = limit_async_func_call(16)(EmbeddingModels.random_embedding)
    bge_embed_func = limit_async_func_call(16)(EmbeddingModels.mock_bge_embedding)
    openai_embed_func = limit_async_func_call(16)(EmbeddingModels.mock_openai_embedding)
    
    # Step 1: Create a LightRAG instance with a dummy embedding function for graph building
    logger.info("Step 1: Creating LightRAG instance with dummy embedding function")
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=random_embed_func,
        llm_model_func=dummy_llm_model_func
    )
    
    # Step 2: Build and serialize the graph without real embedding
    logger.info("Step 2: Building and serializing the graph (without real embedding)")
    # This will build the graph and save it as text for later embedding
    await rag.build_graph_only(SAMPLE_DOCUMENT)
    
    # Step 3: Load the graph and embed it with the BGE model
    logger.info("Step 3: Loading the graph and embedding it with the BGE model")
    await rag.load_graph_and_embed(embedding_func=bge_embed_func, embedding_name="bge")
    
    # Step 4: Load the graph again and embed it with the OpenAI model
    logger.info("Step 4: Loading the graph and embedding it with the OpenAI model")
    await rag.load_graph_and_embed(embedding_func=openai_embed_func, embedding_name="openai")
    
    # Optional: Verify that we have different vector DB files for each embedding model
    bge_entities_file = os.path.join(working_dir, "vdb_entities_bge.json")
    openai_entities_file = os.path.join(working_dir, "vdb_entities_openai.json")
    
    if os.path.exists(bge_entities_file) and os.path.exists(openai_entities_file):
        logger.info("Success! Different vector DB files created for each embedding model:")
        logger.info(f"- BGE entities: {bge_entities_file}")
        logger.info(f"- OpenAI entities: {openai_entities_file}")
    else:
        logger.warning("Expected vector DB files not found!")


if __name__ == "__main__":
    asyncio.run(main()) 
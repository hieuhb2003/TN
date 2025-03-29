import os
import sys
import asyncio
import json
from typing import List, Dict

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# Path to API keys file
API_KEYS_FILE = "api_keys.json"
# Working directory
WORKING_DIR = "./test_duo"

async def main():
    print("Testing entity matching and JSON storage...")
    
    # Make sure working directory exists
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    # Initialize LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "BAAI/bge-m3"
                ),
                embed_model=AutoModel.from_pretrained(
                    "BAAI/bge-m3"
                ),
            ),
        ),
        addon_params={
            "language": "Vietnamese"
        }
    )
    
    # Use a simple LLM function for testing
    async def dummy_llm_func(prompt, **kwargs):
        return "This is a dummy LLM response."
    
    rag.llm_model_func = dummy_llm_func
    
    # Sample entities for testing
    source_entities = [
        "THÔNG TƯ",
        "BỘ TƯ PHÁP",
        "THÔNG TIN CÁ NHÂN"
    ]
    
    target_entities = [
        "CIRCULAR",
        "MINISTRY OF JUSTICE",
        "PERSONAL INFORMATION"
    ]
    
    # Call the matching function directly
    print("Matching entities...")
    matches = await rag._match_entities_embedding(
        source_entities=source_entities,
        target_entities=target_entities,
        source_language="Vietnamese",
        target_language="English",
        similarity_threshold=0.70  # Lower threshold for testing
    )
    
    print(f"Found {len(matches)} matches:")
    for source, target in matches:
        print(f"  {source} <-> {target}")
    
    # Using the utility functions to retrieve and load matched pairs
    print("\nTesting utility functions for matched pairs:")
    
    # 1. Get all matched pairs files
    all_files = rag.get_matched_pairs_files()
    print(f"All matched pairs files ({len(all_files)}):")
    for file in all_files:
        print(f"  {os.path.basename(file)}")
    
    # 2. Filter files by language pair
    vn_en_files = rag.get_matched_pairs_files(source_language="Vietnamese", target_language="English")
    print(f"\nVietnamese-English matched pairs files ({len(vn_en_files)}):")
    for file in vn_en_files:
        print(f"  {os.path.basename(file)}")
    
    # 3. Using the convenience method to get matched pairs directly
    print("\nUsing the convenience method to get matched pairs:")
    
    # Get latest matched pairs
    latest_pairs = rag.get_matched_entity_pairs(
        source_language="Vietnamese", 
        target_language="English",
        latest_only=True
    )
    
    print(f"Latest matched pairs ({len(latest_pairs)}):")
    display_pairs(latest_pairs)
    
    # Get all matched pairs across all files
    all_pairs = rag.get_matched_entity_pairs(
        source_language="Vietnamese", 
        target_language="English",
        latest_only=False
    )
    
    print(f"\nAll matched pairs across files ({len(all_pairs)}):")
    display_pairs(all_pairs)

def display_pairs(pairs):
    """Helper function to display entity pairs"""
    if not pairs:
        print("  No pairs found.")
        return
        
    for i, pair in enumerate(pairs):
        orig_entity = pair["original_entity"]["name"]
        trans_entity = pair["translated_entity"]["name"]
        similarity = pair["similarity_score"]
        
        orig_desc = pair["original_entity"]["description"]
        trans_desc = pair["translated_entity"]["description"]
        
        # Truncate descriptions for display
        orig_desc_short = (orig_desc[:50] + "...") if len(orig_desc) > 50 else orig_desc
        trans_desc_short = (trans_desc[:50] + "...") if len(trans_desc) > 50 else trans_desc
        
        print(f"\n  Pair {i+1} (similarity: {similarity:.3f}):")
        print(f"    Original   ({pair['original_entity']['language']}): {orig_entity}")
        print(f"      Description: {orig_desc_short}")
        print(f"    Translated ({pair['translated_entity']['language']}): {trans_entity}")
        print(f"      Description: {trans_desc_short}")

if __name__ == "__main__":
    asyncio.run(main()) 
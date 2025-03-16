# import os
# import asyncio
# from lightrag import LightRAG, QueryParam
# from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
# from lightrag.kg.shared_storage import initialize_pipeline_status
# import os

# from lightrag import LightRAG, QueryParam
# from lightrag.llm.hf import hf_embed
# from lightrag.utils import EmbeddingFunc
# from transformers import AutoModel, AutoTokenizer
# from lightrag.llm.openai import openai_complete_if_cache

# WORKING_DIR = "./example_benchmark"

# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)
    
# async def llm_model_func(
#     prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
# ) -> str:
#     return await openai_complete_if_cache(
#         os.getenv("LLM_MODEL"),
#         prompt,
#         system_prompt=system_prompt,
#         history_messages=history_messages,
#         api_key=os.getenv("LLM_BINDING_API_KEY"),
#         base_url=os.getenv("LLM_BINDING_HOST"),
#         **kwargs
#     )

# print("Loading model...")
# print(os.getenv("LLM_MODEL"))



# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=llm_model_func,
#     embedding_func=EmbeddingFunc(
#         embedding_dim=1024,
#         max_token_size=5000,
#         func=lambda texts: hf_embed(
#             texts,
#             tokenizer=AutoTokenizer.from_pretrained(
#                 "BAAI/bge-m3"
#             ),
#             embed_model=AutoModel.from_pretrained(
#                 "BAAI/bge-m3"
#             ),
#         ),
#     )
# )

# mode="hybrid"

# res, chunk_list = rag.query(
#     "How can the data from the tracking device be used for transport management and infringement handling?",
#     param=QueryParam(mode=mode,
#                         only_need_context=True,
#                         top_k = 10)
# )

# print(chunk_list)

import os
import asyncio
import json
from typing import List, Dict, Any
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch

WORKING_DIR = "./example_benchmark"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        **kwargs
    )

def init_reranker():
    print("Loading BGE-M3 reranker...")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
    model.eval()
    return model, tokenizer

def rerank_chunks(query: str, chunks: List[str], reranker_model, reranker_tokenizer):
    if not chunks:
        return []
    
    # Prepare pairs for reranking
    pairs = [[query, text] for text in chunks]
    
    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    scores = scores.tolist()
    
    reranked_chunks = [
        {"content": chunk, "rerank_score": score} 
        for chunk, score in zip(chunks, scores)
    ]
    
    reranked_chunks = sorted(reranked_chunks, key=lambda x: x["rerank_score"], reverse=True)
    sorted_contents = [chunk["content"] for chunk in reranked_chunks]

    return sorted_contents

def init_rag():
    print("Initializing LightRAG...")
    print(f"Using LLM model: {os.getenv('LLM_MODEL')}")
    
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3"),
                embed_model=AutoModel.from_pretrained("BAAI/bge-m3"),
            ),
        )
    )

def read_queries_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    
    queries_file = "C:\\Users\\mhieu\\Desktop\\datn\\LIGHTRAG\\example_benchmark\\queries.json" 
    
    rag = init_rag()
    reranker_model, reranker_tokenizer = init_reranker()
    
    # Read queries
    queries = read_queries_from_file(queries_file)
    print(f"Loaded {len(queries)} queries from file")
    
    # Process each query
    results = []
    for query in queries:
        print(f"Processing query: {query}")
        
        # Get chunks using LightRAG
        _, chunk_list = rag.query(
            query,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=10
            )
        )
        
        # Rerank chunks
        reranked_chunks = rerank_chunks(query, chunk_list, reranker_model, reranker_tokenizer)
        
        results.append({query : reranked_chunks})
        
        print(f"Found {len(reranked_chunks)} chunks for query")
    
    # Save results to file
    output_file = "C:\\Users\\mhieu\\Desktop\\datn\\LIGHTRAG\\example_benchmark\\reranked_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
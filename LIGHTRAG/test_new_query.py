import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import json
import time
from typing import List, Dict, Any
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import random
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

WORKING_DIR = "/home/hungpv/projects/TN/LIGHTRAG/set_up_demo_rag"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

with open("/home/hungpv/projects/list_key_open_router/keys.json", 'r', encoding='utf-8') as f:
    OPENROUTER_API_KEYS = json.load(f)

random.shuffle(OPENROUTER_API_KEYS)

class APIManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.failed_keys = set()
        self.last_switch_time = {}  # Để theo dõi thời gian giữa các lần chuyển đổi
        
    def get_current_api_key(self):
        return self.api_keys[self.current_key_index]
    
    def switch_to_next_key(self):
        # Đánh dấu key hiện tại là đã thất bại
        self.failed_keys.add(self.current_key_index)
        self.last_switch_time[self.current_key_index] = time.time()
        
        # Tìm key tiếp theo chưa thất bại hoặc đã qua thời gian chờ
        available_keys = []
        for idx in range(len(self.api_keys)):
            if idx not in self.failed_keys:
                available_keys.append(idx)
            elif idx in self.last_switch_time:
                # Nếu đã qua 10 phút kể từ lần cuối sử dụng key này
                if time.time() - self.last_switch_time[idx] > 600:
                    self.failed_keys.remove(idx)
                    available_keys.append(idx)
        
        if not available_keys:
            raise RuntimeError("All API keys have failed. Please try again later.")
        
        # Chọn một key ngẫu nhiên từ các key có sẵn
        self.current_key_index = random.choice(available_keys)
        return self.get_current_api_key()
    
    def reset_key(self, key_index):
        if key_index in self.failed_keys:
            self.failed_keys.remove(key_index)

# Khởi tạo API Manager
api_manager = APIManager(OPENROUTER_API_KEYS)

# Giữ nguyên hàm bất đồng bộ này
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    max_retries = len(OPENROUTER_API_KEYS)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            current_api_key = api_manager.get_current_api_key()
            print(f"Using API key: {current_api_key[:5]}...")
            
            response = await openai_complete_if_cache(
                "google/gemini-2.0-flash-exp:free",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=current_api_key,
                base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
                **kwargs
            )
            
            # Nếu thành công, đánh dấu key này hoạt động tốt
            api_manager.reset_key(api_manager.current_key_index)
            return response
            
        except Exception as e:
            print(f"Error with API key {api_manager.current_key_index}: {str(e)}")
            retry_count += 1
            
            if retry_count < max_retries:
                print(f"Switching to next API key...")
                api_manager.switch_to_next_key()
            else:
                print("All API keys have failed. Raising error.")
                raise e
    
    raise RuntimeError("All API keys have failed")

# def init_reranker():
#     print("Loading BGE-M3 reranker...")
#     tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
#     model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
#     model.eval()
#     return model, tokenizer


# def rerank_chunks(query: str, chunks: List[str], reranker_model, reranker_tokenizer):
#     if not chunks:
#         return []
    
#     # Prepare pairs for reranking
#     pairs = [[query, text] for text in chunks]
    
#     with torch.no_grad():
#         inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
#         scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
    
#     scores = scores.tolist()
    
#     reranked_chunks = [
#         {"content": chunk, "rerank_score": score} 
#         for chunk, score in zip(chunks, scores)
#     ]
    
#     reranked_chunks = sorted(reranked_chunks, key=lambda x: x["rerank_score"], reverse=True)
#     sorted_contents = [chunk["content"] for chunk in reranked_chunks]

#     return sorted_contents

# Chuyển lại sang hàm đồng bộ
def init_rag():
    print("Initializing LightRAG...")
    print(f"Using LLM model: google/gemini-2.0-flash-exp:free")
    
    # Khởi tạo đối tượng LightRAG
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
    import json 
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def sort_chunk(ll_chunk: List[tuple], hl_chunk: List[tuple]) -> List[tuple]:
    total = ll_chunk + hl_chunk
    reranked_chunks = sorted(total, key=lambda x: x[1], reverse=True) 
    return reranked_chunks

def change_to_list(ls_chunk: List[tuple]):
    return [x[0] for x in ls_chunk]

# Hàm này bây giờ sẽ chạy đồng bộ và sử dụng một event loop riêng
def process_query(rag, query):
    try:
        print(f"Processing query: {query}")
        
        # Get chunks using LightRAG
        ll_chunk_list, hl_chunk_list = rag.retrieval(
            query,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=10
            )
        )
        print("done retrieval")
        # return ll_chunk_list, hl_chunk_list
        combine_list_tuple = sort_chunk(ll_chunk_list, hl_chunk_list)

        print("done sort")
        print(combine_list_tuple)

        if len(combine_list_tuple) == 0:
            print("No chunks found for query")
            item = {
                "LightRAG Hybrid": [],
                "LightRAG Local": [],
                "LightRAG Global": []
            }
            item_dict = {
                "LightRAG Hybrid": combine_list_tuple,
                "LightRAG Local": ll_chunk_list,
                "LightRAG Global": hl_chunk_list               
            }
            return item, item_dict
        
        else:
        
            print(f"Found {len(combine_list_tuple)} chunks for query")

            item_list = {
                "LightRAG Hybrid": change_to_list(combine_list_tuple),
                "LightRAG Local": change_to_list(ll_chunk_list),
                "LightRAG Global": change_to_list(hl_chunk_list)
            }

            item_dict = {
                "LightRAG Hybrid": combine_list_tuple,
                "LightRAG Local": ll_chunk_list,
                "LightRAG Global": hl_chunk_list               
            }

            return item_list, item_dict
    
    except Exception as e:
        print(f"Error processing query '{query}': {str(e)}")
        return {query: [f"ERROR: {str(e)}"]}

def main():
    queries_file = "/home/hungpv/projects/TN/LIGHTRAG/data/small_queries.json"

    rag = init_rag()
    # reranker_model, reranker_tokenizer = init_reranker()
    
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    queries = read_queries_from_file(queries_file)
    queries = queries[:100]
    print(f"Loaded {len(queries)} queries from file")

    # vector_store = FAISS.load_local(
    #     "/home/hungpv/projects/TN/LIGHTRAG/faiss_index", embeddings, allow_dangerous_deserialization=True
    # )

    model_results = {
        "LightRAG Hybrid": {},
        "LightRAG Local": {},
        "LightRAG Global": {},
    }

    model_results_with_score = {
        "LightRAG Hybrid": {},
        "LightRAG Local": {},
        "LightRAG Global": {},
    }
    i = 1
    for query in queries:
        query_result, query_dict = process_query(rag, query)

        
        # results_embedding = vector_store.similarity_search(
        #     query, k=10
        # )

        # docs = [item.page_content for item in results_embedding]
        # # docs = [results_embedding[i].page_content for i in range(10)]

        # model_results["BGE-M3"][query] = docs
        for model_name, chunks in query_result.items():
            print(model_name)
            model_results[model_name][query] = chunks

        for model_name, chunks in query_dict.items():
            print(model_name)
            model_results_with_score[model_name][query] = chunks

        output_file = "/home/hungpv/projects/TN/LIGHTRAG/result_26_3/retrieval_results_true_method.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)

        output_file2 = "/home/hungpv/projects/TN/LIGHTRAG/result_26_3/retrieval_results_with_score_true_method.json"
        with open(output_file2, 'w', encoding='utf-8') as f:
            json.dump(model_results_with_score, f, ensure_ascii=False, indent=2)
        print(i)
        i+=1
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
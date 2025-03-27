import os
import sys
import time
import json
import random
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc, detect_language
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.openai import openai_complete_if_cache

with open("/Users/oraichain/Desktop/rag/TN/LIGHTRAG/api_keys.json", 'r', encoding='utf-8') as f:
    OPENROUTER_API_KEYS = json.load(f)
    
random.shuffle(OPENROUTER_API_KEYS)
# class APIManager:
#     def __init__(self, api_keys: List[str]):
#         self.api_keys = api_keys
#         self.current_key_index = 0
#         self.failed_keys = set()
#         self.last_switch_time = {}  
        
#     def get_current_api_key(self):
#         return self.api_keys[self.current_key_index]
    
#     def switch_to_next_key(self):
        
#         self.failed_keys.add(self.current_key_index)
#         self.last_switch_time[self.current_key_index] = time.time()
        
#         available_keys = []
#         for idx in range(len(self.api_keys)):
#             if idx not in self.failed_keys:
#                 available_keys.append(idx)
#             elif idx in self.last_switch_time:
#                 # Nếu đã qua 10 phút kể từ lần cuối sử dụng key này
#                 if time.time() - self.last_switch_time[idx] > 600:
#                     self.failed_keys.remove(idx)
#                     available_keys.append(idx)
        
#         if not available_keys:
#             raise RuntimeError("Tất cả API keys đều đã thất bại. Vui lòng thử lại sau.")
        
#         self.current_key_index = random.choice(available_keys)
#         print(f"Đã chuyển sang API key: {self.api_keys[self.current_key_index][:5]}...")
#         return self.get_current_api_key()
    
#     def reset_key(self, key_index):
#         if key_index in self.failed_keys:
#             self.failed_keys.remove(key_index)

class APIManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.failed_keys = set()
        self.last_switch_time = {}  
        
    def get_current_api_key(self):
        return self.api_keys[self.current_key_index]
    
    def rotate_to_next_key(self):
        """Rotate to the next available API key after a successful operation"""
        available_keys = []
        for idx in range(len(self.api_keys)):
            if idx not in self.failed_keys:
                available_keys.append(idx)
                
        if not available_keys:
            raise RuntimeError("Tất cả API keys đều đã thất bại. Vui lòng thử lại sau.")
        
        # Find the next available key after the current one
        next_indices = [idx for idx in available_keys if idx > self.current_key_index]
        if next_indices:
            self.current_key_index = min(next_indices)
        else:
            # If there are no available keys with higher indices, wrap around to the beginning
            self.current_key_index = min(available_keys)
            
        print(f"Đã luân chuyển sang API key: {self.api_keys[self.current_key_index][:5]}...")
        return self.get_current_api_key()
    
    def switch_to_next_key(self):
        """Switch to next key after a failure"""
        self.failed_keys.add(self.current_key_index)
        self.last_switch_time[self.current_key_index] = time.time()
        
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
            raise RuntimeError("Tất cả API keys đều đã thất bại. Vui lòng thử lại sau.")
        
        self.current_key_index = random.choice(available_keys)
        print(f"Đã chuyển sang API key: {self.api_keys[self.current_key_index][:5]}...")
        return self.get_current_api_key()
    
    def reset_key(self, key_index):
        if key_index in self.failed_keys:
            self.failed_keys.remove(key_index)

# Khởi tạo API Manager
api_manager = APIManager(OPENROUTER_API_KEYS)

WORKING_DIR = "/Users/oraichain/Desktop/rag/TN/LIGHTRAG/set_up_demo_rag"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

current_api_key = api_manager.get_current_api_key()
    
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    global current_api_key
    max_retries = len(OPENROUTER_API_KEYS)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = await openai_complete_if_cache(
                "google/gemini-2.0-pro-exp-02-05:free",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=current_api_key, 
                base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
                **kwargs
            )
            
            key_index = OPENROUTER_API_KEYS.index(current_api_key)
            api_manager.reset_key(key_index)
            return response
            
        except Exception as e:
            print(f"Lỗi với API key: {str(e)}")
            retry_count += 1
            
            if retry_count < max_retries:
                print("Đang chuyển sang API key tiếp theo...")
                current_api_key = api_manager.switch_to_next_key()
                os.environ["LLM_BINDING_API_KEY"] = current_api_key
            else:
                print("Tất cả API keys đều đã thất bại.")
                raise e
    
    raise RuntimeError("Tất cả API keys đều đã thất bại")

print("Loading model...")
print("google/gemini-2.0-pro-exp-02-05:free")

os.environ["LLM_BINDING_API_KEY"] = current_api_key

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
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
        "insert_batch_size": 20  
    }
)

def insert_with_retry(data, language="Vietnamese"):
    global current_api_key
    max_retries = len(OPENROUTER_API_KEYS)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            os.environ["LLM_BINDING_API_KEY"] = current_api_key
            rag.insert(data, language=language, matching_method="None")
            # rag.insert_duo(data_original=data[0], data_translated=data[1])
            print("Chèn dữ liệu thành công!")
            
            # Luân chuyển sang API key tiếp theo sau mỗi lần insert thành công
            current_api_key = api_manager.rotate_to_next_key()
            return
            
        except Exception as e:
            error_str = str(e).lower()
            if "api" in error_str or "key" in error_str or "rate" in error_str or "limit" in error_str:
                print(f"Lỗi API khi chèn dữ liệu: {str(e)}")
                retry_count += 1
                
                if retry_count < max_retries:
                    print("Đang chuyển sang API key tiếp theo...")
                    current_api_key = api_manager.switch_to_next_key()
                    os.environ["LLM_BINDING_API_KEY"] = current_api_key
                else:
                    print("Tất cả API keys đều đã thất bại khi chèn dữ liệu.")
                    raise e
            else:
                raise e

def main():
    try:
        with open("/Users/oraichain/Desktop/rag/TN/LIGHTRAG/data/context.json", 'r') as f:
            data = json.load(f)
        from tqdm import tqdm
        for i, item in tqdm(enumerate(data)):
            language = detect_language(item)    
            insert_with_retry(item, language=language)
        
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {str(e)}")

if __name__ == "__main__":
    main()
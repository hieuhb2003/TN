import os
import sys
import time
import json
import random
from typing import List
from dotenv import load_dotenv
import argparse  # Thêm thư viện argparse

load_dotenv()

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import detect_language

# WORKING_DIR = "./duo_graph"
    
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
 
    max_retries = 10
    retry_count = 0
    model = os.getenv("LLM_MODEL")
    while retry_count < max_retries:
        try:
            response = await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("LLM_BINDING_API_KEY"), 
                base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
                **kwargs
            )
            
            return response
            
        except Exception as e:
            print(f"Lỗi tại lần: {str(retry_count)}/{str(max_retries)}")
            retry_count += 1
            
    
    raise RuntimeError("Tất cả API keys đều đã thất bại")

print(f"Loading model...{os.getenv('LLM_MODEL')}")

# Hàm tạo đối tượng rag với các tham số truyền vào
def create_rag(working_dir, model_func):
    return LightRAG(
        working_dir=working_dir,
        llm_model_func=model_func,
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

# Thêm hàm để xử lý đối số từ dòng lệnh
def parse_arguments():
    parser = argparse.ArgumentParser(description="Chương trình xử lý dữ liệu với LightRAG")
    parser.add_argument("--working_dir", type=str, default="./duo_graph", help="Đường dẫn tới thư mục làm việc")
    parser.add_argument("--data_path", type=str, required=True, help="Đường dẫn tới file dữ liệu JSON")
    return parser.parse_args()

def insert_with_retry(rag, data_original, data_translated, target_language="English"):

    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            rag.insert_duo(data_original, data_translated, target_language="English")
            print("Chèn dữ liệu thành công!")
            return
        except Exception as e:
            print(f"Lỗi khi chèn dữ liệu: {str(e)}")
            retry_count += 1
            print(f"Đang thử lại lần thứ {retry_count+1}/{max_retries}...")

def main():
    args = parse_arguments()  # Lấy đối số từ dòng lệnh

    try:
        rag = create_rag(args.working_dir, llm_model_func)  # Tạo đối tượng rag từ tham số truyền vào
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            insert_with_retry(rag, item[0], item[1], target_language="English")
        
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {str(e)}")

if __name__ == "__main__":
    main()

# import os
# import sys
# import time
# import json
# import random
# import asyncio
# from typing import List


# from lightrag import LightRAG, QueryParam
# from lightrag.llm.hf import hf_embed
# from lightrag.utils import EmbeddingFunc
# from transformers import AutoModel, AutoTokenizer
# from lightrag.llm.openai import openai_complete_if_cache
# from lightrag.utils import detect_language

# # Đọc file API keys
# with open("C:/Users/mhieu/Desktop/TN/LIGHTRAG/api_keys.json", 'r', encoding='utf-8') as f:
#     OPENROUTER_API_KEYS = json.load(f)

# # OPENROUTER_API_KEYS = OPENROUTER_API_KEYS[1000:] #700 key
# # with open("/home/hungpv/projects/list_key_open_router/openrouter_full_3_new.json", 'r', encoding='utf-8') as f:
# #     OPENROUTER_API_KEYS_2 = json.load(f)
# # OPENROUTER_API_KEYS = OPENROUTER_API_KEYS + OPENROUTER_API_KEYS_2
# # OPENROUTER_API_KEYS = OPENROUTER_API_KEYS[:int(len(OPENROUTER_API_KEYS)/2)]
# print(f"Tổng số API keys: {len(OPENROUTER_API_KEYS)}")
# random.shuffle(OPENROUTER_API_KEYS)

# # Thêm lock để xử lý đồng thời
# class APIManager:
#     def __init__(self, api_keys: List[str]):
#         self.api_keys = api_keys
#         self.current_key_index = 0
#         self.failed_keys = set()
#         self.last_switch_time = {}
#         self.lock = asyncio.Lock()  # Thêm lock để xử lý đồng thời
        
#     async def get_current_api_key(self):
#         async with self.lock:
#             return self.api_keys[self.current_key_index]
    
#     async def switch_to_next_key(self, mark_current_as_failed=False):
#         async with self.lock:
#             # Chỉ đánh dấu key là failed nếu có yêu cầu
#             if mark_current_as_failed:
#                 self.failed_keys.add(self.current_key_index)
#                 self.last_switch_time[self.current_key_index] = time.time()
            
#             # Lưu index hiện tại để không chọn lại key đó
#             previous_index = self.current_key_index
            
#             # Cập nhật danh sách available keys
#             available_keys = []
#             for idx in range(len(self.api_keys)):
#                 # Không chọn lại key hiện tại và các key đã thất bại
#                 if idx != previous_index and idx not in self.failed_keys:
#                     available_keys.append(idx)
#                 # Phục hồi key đã qua thời gian chờ
#                 elif idx in self.last_switch_time and idx != previous_index:
#                     if time.time() - self.last_switch_time[idx] > 600:  # 10 phút
#                         if idx in self.failed_keys:
#                             self.failed_keys.remove(idx)
#                         available_keys.append(idx)
            
#             if not available_keys:
#                 # Nếu không còn key nào, quay về key đầu tiên và đánh dấu tất cả keys đã reset
#                 self.failed_keys.clear()
#                 self.last_switch_time.clear()
#                 available_keys = list(range(len(self.api_keys)))
#                 if previous_index in available_keys:
#                     available_keys.remove(previous_index)
            
#             # Chọn một key ngẫu nhiên từ danh sách khả dụng
#             self.current_key_index = random.choice(available_keys)
#             print(f"Đã chuyển sang API key: {self.api_keys[self.current_key_index][:5]}...")
#             return self.api_keys[self.current_key_index]
    
#     async def mark_key_failed(self, key_index):
#         async with self.lock:
#             self.failed_keys.add(key_index)
#             self.last_switch_time[key_index] = time.time()

# # Khởi tạo API Manager
# api_manager = APIManager(OPENROUTER_API_KEYS)

# WORKING_DIR = "./zalo_graph_single_vi_without_embedding"

# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)

# # Sử dụng biến global cho API key hiện tại
# current_api_key = OPENROUTER_API_KEYS[0]  # Khởi tạo với key đầu tiên

# # Semaphore để giới hạn số lượng request đồng thời
# api_semaphore = asyncio.Semaphore(5)  # Giới hạn 5 requests đồng thời

# async def llm_model_func(
#     prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
# ) -> str:
#     global current_api_key
#     max_retries = min(len(OPENROUTER_API_KEYS), 20)  # Giới hạn số lần retry
#     retry_count = 0
    
#     # Sử dụng semaphore để giới hạn số lượng request đồng thời
#     async with api_semaphore:
#         # Luôn chuyển sang key mới trước khi gọi API
#         current_api_key = await api_manager.switch_to_next_key(mark_current_as_failed=False)
#         os.environ["LLM_BINDING_API_KEY"] = current_api_key
        
#         while retry_count < max_retries:
#             try:
#                 response = await openai_complete_if_cache(
#                     "google/gemini-2.0-flash-exp:free",
#                     prompt,
#                     system_prompt=system_prompt,
#                     history_messages=history_messages,
#                     api_key=current_api_key, 
#                     base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
#                     **kwargs
#                 )
                
#                 # Không đánh dấu key là failed khi thành công
#                 return response
                
#             except Exception as e:
#                 error_str = str(e).lower()
#                 print(f"Lỗi với API key {current_api_key[:5]}: {str(e)}")
                
#                 # Đánh dấu key hiện tại là failed
#                 current_key_index = OPENROUTER_API_KEYS.index(current_api_key)
#                 await api_manager.mark_key_failed(current_key_index)
                
#                 retry_count += 1
#                 if retry_count < max_retries:
#                     # Thêm một khoảng chờ nhỏ trước khi thử lại
#                     await asyncio.sleep(0.5 * retry_count)  # Tăng dần thời gian chờ
#                     print(f"Thử lại lần {retry_count}/{max_retries}...")
#                     current_api_key = await api_manager.switch_to_next_key(mark_current_as_failed=False)
#                     os.environ["LLM_BINDING_API_KEY"] = current_api_key
#                 else:
#                     print("Đã vượt quá số lần thử lại.")
#                     raise RuntimeError(f"Không thể hoàn thành yêu cầu sau {max_retries} lần thử: {str(e)}")
    
#     raise RuntimeError("Tất cả API keys đều đã thất bại")

# print("Loading model...")
# print("meta-llama/llama-3.3-70b-instruct:free")

# # Khởi tạo API key ban đầu
# os.environ["LLM_BINDING_API_KEY"] = current_api_key

# # Khởi tạo LightRAG
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
#     ),
#     addon_params={
#         "insert_batch_size": 5,  # Giảm kích thước batch để giảm áp lực lên API
#         "language": "Vietnamese"
#     }
# )

# # Sửa lại hàm insert_with_retry để tránh lỗi với event loop
# def insert_with_retry(data, language):
#     global current_api_key
#     max_retries = min(len(OPENROUTER_API_KEYS), 20)
#     retry_count = 0
    
#     while retry_count < max_retries:
#         try:
#             # Không tạo event loop mới, chỉ thay đổi API key
#             current_api_key = OPENROUTER_API_KEYS[(OPENROUTER_API_KEYS.index(current_api_key) + 1) % len(OPENROUTER_API_KEYS)]
#             os.environ["LLM_BINDING_API_KEY"] = current_api_key
            
#             # Gọi hàm insert đồng bộ, không phải async
#             rag.insert(
#                 data,
#                 language=language,
#                 matching_method="embedding", 
#                 delay_vector_db_update=True, 
#                 need_cross_language=False
#             )
#             print("Chèn dữ liệu thành công!")
#             return
#         except Exception as e:
#             error_str = str(e).lower()
#             print(f"Lỗi khi chèn dữ liệu với key {current_api_key[:5]}: {str(e)}")
            
#             # Thay đổi key khi gặp lỗi
#             retry_count += 1
#             if retry_count < max_retries:
#                 # Đơn giản hóa việc chuyển key để tránh lỗi event loop
#                 current_key_index = OPENROUTER_API_KEYS.index(current_api_key)
#                 current_api_key = OPENROUTER_API_KEYS[(current_key_index + 1) % len(OPENROUTER_API_KEYS)]
#                 os.environ["LLM_BINDING_API_KEY"] = current_api_key
#                 print(f"Đã chuyển sang API key: {current_api_key[:5]}...")
#                 print(f"Thử lại insert lần {retry_count}/{max_retries}...")
#             else:
#                 print("Đã vượt quá số lần thử lại khi chèn dữ liệu.")
#                 raise RuntimeError(f"Không thể chèn dữ liệu sau {max_retries} lần thử: {str(e)}")

# def main():
#     try:
#         with open("C:/Users/mhieu/Desktop/TN/LIGHTRAG/data/single_data.json", 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         lang = detect_language(data[0])
        
#         data_batch = []
#         batch_size = 5
#         for i in range(0, len(data), batch_size):
#             data_batch.append(data[i:i+batch_size])
        
#         from tqdm import tqdm
#         for item in tqdm(data_batch):
#             print(f"Ngôn ngữ phát hiện: {lang}")
#             print(f"Xử lý batch với {len(item)} mục")
#             insert_with_retry(item, language=lang)
#             # Sử dụng sleep thông thường thay vì asyncio.sleep
#             time.sleep(2)
#     except Exception as e:
#         print(f"Lỗi khi xử lý dữ liệu: {str(e)}")

# # Không cần gọi event loop nữa
# if __name__ == "__main__":
#     main()

import os
import sys
import time
import json
import random
import asyncio
from typing import List


from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import detect_language

# Đọc file API keys
with open("/home/hungpv/projects/list_key_open_router/openrouter_full_2_new.json", 'r', encoding='utf-8') as f:
    OPENROUTER_API_KEYS = json.load(f)
# OPENROUTER_API_KEYS = OPENROUTER_API_KEYS[:750]
# OPENROUTER_API_KEYS = OPENROUTER_API_KEYS[1000:] #700 key
# with open("/home/hungpv/projects/list_key_open_router/openrouter_full_3_new.json", 'r', encoding='utf-8') as f:
#     OPENROUTER_API_KEYS_2 = json.load(f)
# OPENROUTER_API_KEYS = OPENROUTER_API_KEYS + OPENROUTER_API_KEYS_2
# OPENROUTER_API_KEYS = OPENROUTER_API_KEYS[:int(len(OPENROUTER_API_KEYS)/2)]
print(f"Tổng số API keys: {len(OPENROUTER_API_KEYS)}")
# random.shuffle(OPENROUTER_API_KEYS)

# Thêm lock để xử lý đồng thời
class APIManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.failed_keys = set()
        self.permanently_banned_keys = set()  # Thêm set để lưu các key bị cấm vĩnh viễn do lỗi 429
        self.last_switch_time = {}
        self.lock = asyncio.Lock()  # Thêm lock để xử lý đồng thời
        
    async def get_current_api_key(self):
        async with self.lock:
            return self.api_keys[self.current_key_index]
    
    async def switch_to_next_key(self, mark_current_as_failed=False):
        async with self.lock:
            # Chỉ đánh dấu key là failed nếu có yêu cầu
            if mark_current_as_failed:
                self.failed_keys.add(self.current_key_index)
                self.last_switch_time[self.current_key_index] = time.time()
            
            # Lưu index hiện tại để không chọn lại key đó
            previous_index = self.current_key_index
            
            # Cập nhật danh sách available keys
            available_keys = []
            for idx in range(len(self.api_keys)):
                # Không chọn lại key hiện tại, các key đã thất bại tạm thời, và các key bị cấm vĩnh viễn
                if (idx != previous_index and 
                    idx not in self.failed_keys and 
                    idx not in self.permanently_banned_keys):
                    available_keys.append(idx)
                # Phục hồi key đã qua thời gian chờ (chỉ cho key thất bại tạm thời, không cho key bị cấm vĩnh viễn)
                elif idx in self.last_switch_time and idx != previous_index and idx not in self.permanently_banned_keys:
                    if time.time() - self.last_switch_time[idx] > 600:  # 10 phút
                        if idx in self.failed_keys:
                            self.failed_keys.remove(idx)
                        available_keys.append(idx)
            
            # Nếu không còn key nào khả dụng, dừng chương trình
            if not available_keys:
                print("CẢNH BÁO: Không còn API key nào khả dụng!")
                print(f"- Tổng số key: {len(self.api_keys)}")
                print(f"- Key thất bại tạm thời: {len(self.failed_keys)}")
                print(f"- Key bị cấm vĩnh viễn (429): {len(self.permanently_banned_keys)}")
                raise RuntimeError("Không còn API key khả dụng. Chương trình buộc phải dừng lại.")
            
            # Chọn một key ngẫu nhiên từ danh sách khả dụng
            self.current_key_index = random.choice(available_keys)
            print(f"Đã chuyển sang API key: {self.api_keys[self.current_key_index][:5]}...")
            return self.api_keys[self.current_key_index]
    
    async def mark_key_failed(self, key_index):
        async with self.lock:
            self.failed_keys.add(key_index)
            self.last_switch_time[key_index] = time.time()
            
    async def mark_key_rate_limited(self, key_index):
        """Đánh dấu key bị cấm vĩnh viễn do lỗi rate limit (429)"""
        async with self.lock:
            self.permanently_banned_keys.add(key_index)
            # Xóa khỏi danh sách key thất bại tạm thời nếu có
            if key_index in self.failed_keys:
                self.failed_keys.remove(key_index)
            print(f"API key {self.api_keys[key_index][:5]} đã bị đánh dấu là cấm vĩnh viễn do lỗi 429")

# Khởi tạo API Manager
api_manager = APIManager(OPENROUTER_API_KEYS)

WORKING_DIR = "./new_zalo_graph_single_en_without_embedding"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Sử dụng biến global cho API key hiện tại
current_api_key = OPENROUTER_API_KEYS[0]  # Khởi tạo với key đầu tiên

# Semaphore để giới hạn số lượng request đồng thời
api_semaphore = asyncio.Semaphore(5)  # Giới hạn 5 requests đồng thời

# Cập nhật hàm llm_model_func để xử lý lỗi 429
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    global current_api_key
    max_retries = min(len(OPENROUTER_API_KEYS), 30)  # Giới hạn số lần retry
    retry_count = 0
    
    # Định nghĩa các chuỗi để nhận diện lỗi rate limit
    rate_limit_indicators = [
        "429",
        "too many requests", 
        "rate limit", 
        "quota exceeded", 
        "resource_exhausted",
        "provider returned error"  # OpenRouter thường trả về lỗi này khi có lỗi 429
    ]
    
    # Sử dụng semaphore để giới hạn số lượng request đồng thời
    async with api_semaphore:
        # Luôn chuyển sang key mới trước khi gọi API
        current_api_key = await api_manager.switch_to_next_key(mark_current_as_failed=False)
        os.environ["LLM_BINDING_API_KEY"] = current_api_key
        
        while retry_count < max_retries:
            try:
                response = await openai_complete_if_cache(
                    "qwen/qwen-2.5-7b-instruct:free",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=current_api_key, 
                    base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
                    **kwargs
                )
                
                # Kiểm tra xem response có chứa error không (đặc biệt là lỗi 429)
                if hasattr(response, 'error') and response.error is not None:
                    error_info = response.error
                    error_str = str(error_info)
                    print(f"Lỗi trong response: {error_str}")
                    
                    # Xác định xem có phải lỗi 429 không
                    is_rate_limit = False
                    if isinstance(error_info, dict) and 'code' in error_info and error_info['code'] == 429:
                        is_rate_limit = True
                    else:
                        error_lower = error_str.lower()
                        is_rate_limit = any(indicator in error_lower for indicator in rate_limit_indicators)
                    
                    if is_rate_limit:
                        # Nếu là lỗi 429, đánh dấu key này bị cấm vĩnh viễn
                        current_key_index = OPENROUTER_API_KEYS.index(current_api_key)
                        await api_manager.mark_key_rate_limited(current_key_index)
                        
                        # Thử với key khác
                        retry_count += 1
                        if retry_count < max_retries:
                            await asyncio.sleep(0.5 * retry_count)
                            print(f"Thử lại lần {retry_count}/{max_retries}...")
                            current_api_key = await api_manager.switch_to_next_key(mark_current_as_failed=False)
                            os.environ["LLM_BINDING_API_KEY"] = current_api_key
                            continue
                        else:
                            raise RuntimeError(f"Không thể hoàn thành yêu cầu sau {max_retries} lần thử")
                
                # Không đánh dấu key là failed khi thành công
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                print(f"Lỗi với API key {current_api_key[:5]}: {str(e)}")
                
                # Kiểm tra xem có phải lỗi rate limit không
                is_rate_limit = any(indicator in error_str for indicator in rate_limit_indicators)
                
                # Lấy index của key hiện tại
                current_key_index = OPENROUTER_API_KEYS.index(current_api_key)
                
                if is_rate_limit:
                    # Nếu là lỗi rate limit, đánh dấu key này bị cấm vĩnh viễn
                    print(f"Phát hiện lỗi rate limit! API key {current_api_key[:5]} bị đánh dấu cấm vĩnh viễn.")
                    await api_manager.mark_key_rate_limited(current_key_index)
                else:
                    # Nếu không phải lỗi rate limit, đánh dấu key này thất bại tạm thời
                    await api_manager.mark_key_failed(current_key_index)
                
                retry_count += 1
                if retry_count < max_retries:
                    # Thêm một khoảng chờ nhỏ trước khi thử lại
                    await asyncio.sleep(0.5 * retry_count)  # Tăng dần thời gian chờ
                    print(f"Thử lại lần {retry_count}/{max_retries}...")
                    try:
                        current_api_key = await api_manager.switch_to_next_key(mark_current_as_failed=False)
                        os.environ["LLM_BINDING_API_KEY"] = current_api_key
                    except RuntimeError as e:
                        # Nếu không còn key nào khả dụng, dừng chương trình
                        print(str(e))
                        sys.exit(1)
                else:
                    print("Đã vượt quá số lần thử lại.")
                    raise RuntimeError(f"Không thể hoàn thành yêu cầu sau {max_retries} lần thử: {str(e)}")
    
    raise RuntimeError("Tất cả API keys đều đã thất bại")

print("Loading model...")
print("qwen/qwen-2.5-7b-instruct:free")

# Khởi tạo API key ban đầu
os.environ["LLM_BINDING_API_KEY"] = current_api_key

# Khởi tạo LightRAG
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
        "insert_batch_size": 5,  # Giảm kích thước batch để giảm áp lực lên API
        # "language": "Vietnamese"
    }
)

# Cập nhật hàm insert_with_retry để xử lý lỗi 429 và dừng khi không còn key
def insert_with_retry(data, language):
    global current_api_key
    max_retries = min(len(OPENROUTER_API_KEYS), 20)
    retry_count = 0
    
    # Định nghĩa các chuỗi để nhận diện lỗi rate limit
    rate_limit_indicators = [
        "429",
        "too many requests", 
        "rate limit", 
        "quota exceeded", 
        "resource_exhausted",
        "provider returned error"
    ]
    
    while retry_count < max_retries:
        try:
            # Kiểm tra có còn key khả dụng không
            available_keys = [
                idx for idx in range(len(OPENROUTER_API_KEYS)) 
                if idx not in api_manager.permanently_banned_keys and idx not in api_manager.failed_keys
            ]
            
            if not available_keys:
                print("CẢNH BÁO: Không còn API key nào khả dụng!")
                print(f"- Tổng số key: {len(OPENROUTER_API_KEYS)}")
                print(f"- Key thất bại tạm thời: {len(api_manager.failed_keys)}")
                print(f"- Key bị cấm vĩnh viễn (429): {len(api_manager.permanently_banned_keys)}")
                sys.exit(1)  # Dừng chương trình
            
            # Chọn một key ngẫu nhiên từ các key khả dụng
            next_key_index = random.choice(available_keys)
            current_api_key = OPENROUTER_API_KEYS[next_key_index]
            os.environ["LLM_BINDING_API_KEY"] = current_api_key
            print(f"Đã chuyển sang API key: {current_api_key[:5]}...")
            
            # Gọi hàm insert đồng bộ
            rag.insert(
                data,
                language=language,
                matching_method="embedding", 
                delay_vector_db_update=True, 
                need_cross_language=False
            )
            print("Chèn dữ liệu thành công!")
            return
        except Exception as e:
            error_str = str(e).lower()
            print(f"Lỗi khi chèn dữ liệu với key {current_api_key[:5]}: {str(e)}")
            
            # Kiểm tra xem có phải lỗi rate limit không
            is_rate_limit = any(indicator in error_str for indicator in rate_limit_indicators)
            
            # Lấy index của key hiện tại
            current_key_index = OPENROUTER_API_KEYS.index(current_api_key)
            
            if is_rate_limit:
                # Nếu là lỗi rate limit, đánh dấu key này bị cấm vĩnh viễn
                print(f"Phát hiện lỗi rate limit! API key {current_api_key[:5]} bị đánh dấu cấm vĩnh viễn.")
                api_manager.permanently_banned_keys.add(current_key_index)
                if current_key_index in api_manager.failed_keys:
                    api_manager.failed_keys.remove(current_key_index)
            else:
                # Nếu không phải lỗi rate limit, đánh dấu key này thất bại tạm thời
                api_manager.failed_keys.add(current_key_index)
                api_manager.last_switch_time[current_key_index] = time.time()
            
            # Thay đổi key khi gặp lỗi
            retry_count += 1
            if retry_count < max_retries:
                # Đợi một chút trước khi thử lại
                time.sleep(1 * retry_count)  # Tăng dần thời gian chờ
                print(f"Thử lại insert lần {retry_count}/{max_retries}...")
            else:
                print("Đã vượt quá số lần thử lại khi chèn dữ liệu.")
                raise RuntimeError(f"Không thể chèn dữ liệu sau {max_retries} lần thử: {str(e)}")

def main():
    try:
        with open("/home/hungpv/projects/TN/data/list_duo_corpus/data_zalo_legal.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = [item[1] for item in data]
        lang = detect_language(data[0])
        
        data_batch = []
        batch_size = 5
        for i in range(0, len(data), batch_size):
            data_batch.append(data[i:i+batch_size])
        
        from tqdm import tqdm
        for item in tqdm(data_batch):
            print(f"Ngôn ngữ phát hiện: {lang}")
            print(f"Xử lý batch với {len(item)} mục")
            insert_with_retry(item, language=lang)
            # Sử dụng sleep thông thường thay vì asyncio.sleep
            time.sleep(2)
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {str(e)}")

# Không cần gọi event loop nữa
if __name__ == "__main__":
    main()
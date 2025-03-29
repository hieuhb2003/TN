import os
import sys
import json
import asyncio
from typing import List

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.openai import openai_complete_if_cache

# Import our improved implementation
from new_improved_lightrag_duo import install_improved_ainsert_duo

# Load API keys (adjust path as needed)
with open("api_keys.json", 'r', encoding='utf-8') as f:
    API_KEYS = json.load(f)

WORKING_DIR = "./test_improved_duo"

# Set up LLM function
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    api_key = API_KEYS[0]  # Use first API key
    try:
        response = await openai_complete_if_cache(
            "google/gemini-1.5-pro-latest",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key, 
            base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
            **kwargs
        )
        return response
    except Exception as e:
        print(f"Error with API: {str(e)}")
        raise e

# Initialize LightRAG
print("Initializing LightRAG...")
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
        "language": "Vietnamese"
    }
)

# Install the improved ainsert_duo method
print("Installing improved duo insertion...")
install_improved_ainsert_duo()

async def test_improved_duo_insertion():
    print("Testing improved duo insertion...")
    
    # Sample Vietnamese text
    vietnamese_text = """
    Công nghệ trí tuệ nhân tạo đang phát triển nhanh chóng trong những năm gần đây. 
    Các mô hình ngôn ngữ lớn như GPT-4, Claude và Gemini đã đạt được những tiến bộ đáng kể. 
    OpenAI là một trong những công ty hàng đầu trong lĩnh vực AI. 
    Họ đã phát triển ChatGPT, một ứng dụng trò chuyện AI phổ biến. 
    Google cũng đã phát triển mô hình Gemini của riêng mình. 
    Các công nghệ này đang được áp dụng trong nhiều lĩnh vực khác nhau như y tế, giáo dục và tài chính.
    """
    
    # Sample English translation
    english_text = """
    Artificial intelligence technology has been developing rapidly in recent years.
    Large language models like GPT-4, Claude, and Gemini have made significant progress.
    OpenAI is one of the leading companies in the AI field.
    They have developed ChatGPT, a popular AI chat application.
    Google has also developed its own Gemini model.
    These technologies are being applied in various fields such as healthcare, education, and finance.
    """
    
    try:
        # Insert with both texts provided
        print("Inserting documents with both languages provided...")
        original_id, translated_id = await rag.ainsert_duo(
            vietnamese_text, 
            english_text,
            source_language="Vietnamese",
            target_language="English"
        )
        print(f"Inserted documents with IDs: {original_id}, {translated_id}")
        
        # Test with only original text (will generate translation)
        print("\nInsert with only original text (translation will be generated)...")
        test_text = """
        Biến đổi khí hậu là một trong những thách thức lớn nhất của thời đại chúng ta.
        Nhiệt độ trái đất đang tăng lên do hoạt động của con người.
        Các nhà khoa học ở Liên Hợp Quốc đã cảnh báo rằng chúng ta chỉ còn một khoảng thời gian ngắn để ngăn chặn những hậu quả tồi tệ nhất.
        Nhiều quốc gia đã cam kết giảm lượng khí thải carbon.
        Việt Nam cũng đã đặt mục tiêu đạt trung hòa carbon vào năm 2050.
        """
        
        orig_id, trans_id = await rag.ainsert_duo(
            test_text,
            None,  # No translation provided
            source_language="Vietnamese",
            target_language="English"
        )
        print(f"Inserted documents with IDs: {orig_id}, {trans_id}")
        
        # Get the entities from the knowledge graph
        entities = []
        for node_id, data in rag.chunk_entity_relation_graph._graph.nodes(data=True):
            entity_data = {
                "name": node_id,
                "type": data.get("entity_type", ""),
                "description": data.get("description", ""),
                "language": data.get("language", "")
            }
            entities.append(entity_data)
        
        print(f"\nExtracted {len(entities)} entities:")
        for i, entity in enumerate(entities[:10]):  # Show first 10
            print(f"{i+1}. {entity['name']} ({entity['language']}): {entity['description'][:50]}...")
        
        # Get the relationships from the knowledge graph
        relationships = []
        for src, tgt, data in rag.chunk_entity_relation_graph._graph.edges(data=True):
            rel_data = {
                "source": src,
                "target": tgt,
                "description": data.get("description", ""),
                "language": data.get("language", "")
            }
            relationships.append(rel_data)
        
        print(f"\nExtracted {len(relationships)} relationships:")
        for i, rel in enumerate(relationships[:10]):  # Show first 10
            print(f"{i+1}. {rel['source']} → {rel['target']} ({rel['language']}): {rel['description'][:50]}...")
        
        # Test querying in both languages
        print("\nTesting query in Vietnamese...")
        vn_results = await rag.aquery(
            "OpenAI đã phát triển những công nghệ gì?",
            QueryParam(mode="hybrid")
        )
        print("Vietnamese query results:", vn_results)
        
        print("\nTesting query in English...")
        en_results = await rag.aquery(
            "What technologies has OpenAI developed?",
            QueryParam(mode="hybrid")
        )
        print("English query results:", en_results)
        
        return True
    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    try:
        # Create event loop and run the async test
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        success = loop.run_until_complete(test_improved_duo_insertion())
        
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed!")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
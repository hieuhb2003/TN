import os
import re
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from tqdm import tqdm
import json
from dotenv import load_dotenv
load_dotenv()
WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete,
)

with open("/Users/oraichain/Desktop/DATN/LightRAG/queries.txt", "r") as f:
    data = f.read()

queries = re.findall(r"- Question \d+: (.+)", data)
queries = queries[:10]
print(f"Loaded {len(queries)} queries")

modes = ["local", "global", "hybrid", "mix"]

all_results = {}

for mode in modes:
    print(f"\nProcessing queries using {mode.upper()} mode...")
    results = []

    for query in tqdm(queries, desc=f"LightRAG {mode} Processing"):

        response = rag.query(
            query,
            param=QueryParam(mode=mode)
        )

        results.append({
            "query": query,
            "answer": response,
            "mode": mode
        })
    
    all_results[mode] = results
    
    with open(f"/Users/oraichain/Desktop/DATN/lightrag_{mode}_results.json", "w") as f:
        json.dump(results, f, indent=4)
        print(f"Results for {mode} mode saved to lightrag_{mode}_results.json")

with open("/Users/oraichain/Desktop/DATN/lightrag_all_results.json", "w") as f:
    json.dump(all_results, f, indent=4)
    print("All results saved to lightrag_all_results.json")
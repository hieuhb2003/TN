import re
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
load_dotenv()

def direct_eval(query_file, result1_file, result2_file, output_file):
    client = OpenAI()

    with open(query_file, "r") as f:
        data = f.read()
    queries = re.findall(r"- Question \d+: (.+)", data)
    queries = queries[:10]
    print(f"Loaded {len(queries)} queries")

    with open(result1_file, "r") as f:
        result1_data = json.load(f)
    
    with open(result2_file, "r") as f:
        result2_data = json.load(f)
    
    if "result" in result1_data:
        # Single result object with a "result" list
        answers1 = [item["answer"] for item in result1_data["result"]]
    elif isinstance(result1_data, list) and "result" in result1_data[0]:
        # List of objects with "result" key
        answers1 = [item["result"] for item in result1_data]
    elif isinstance(result1_data, list) and "answer" in result1_data[0]:
        # List of objects with "answer" key
        answers1 = [item["answer"] for item in result1_data]
    else:
        raise ValueError(f"Unknown format in {result1_file}")
    
    # Similar extraction for the second result file
    if "result" in result2_data:
        # Single result object with a "result" list
        answers2 = [item["answer"] for item in result2_data["result"]]
    elif isinstance(result2_data, list) and "result" in result2_data[0]:
        # List of objects with "result" key
        answers2 = [item["result"] for item in result2_data]
    elif isinstance(result2_data, list) and "answer" in result2_data[0]:
        # List of objects with "answer" key
        answers2 = [item["answer"] for item in result2_data]
    else:
        raise ValueError(f"Unknown format in {result2_file}")
    

    all_results = []
    sys_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
    """

    for i, (query, answer1, answer2) in enumerate(tqdm(zip(queries, answers1, answers2), total=len(queries))):
        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content

            try:
                content = content.replace("```json", "").replace("```", "").strip()
                content = content.encode('utf-8').decode('unicode_escape')
                content = content.replace("\u2019", "'") 
                content = content.replace("\u201c", "\"").replace("\u201d", "\"") 
                content = content.replace("\u00e2\u0080\u0099", "'") 
                eval_data = json.loads(content, strict=False)

                result = {
                    "query_id": i + 1,
                    "query": query,
                    "evaluation": eval_data
                }
                
                all_results.append(result)

                winner = eval_data.get("Overall Winner", {}).get("Winner", "Unknown")
                print(f"Query {i+1}: Overall winner - {winner}")
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON response for query {i+1}")
                all_results.append({
                    "query_id": i + 1,
                    "query": query,
                    "raw_response": content,
                    "error": "JSON parse error"
                })
                
        except Exception as e:
            print(f"Error processing query {i+1}: {str(e)}")
            all_results.append({
                "query_id": i + 1,
                "query": query,
                "error": str(e)
            })

            time.sleep(1)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Evaluation results saved to {output_file}")
    
    summarize_direct_evaluation(all_results, output_file.replace(".json", "_summary.json"))
    
    return all_results

def summarize_direct_evaluation(evaluation_results, summary_file):
    """
    Summarize evaluation results from direct evaluations
    
    Args:
        evaluation_results: List of evaluation results
        summary_file: File to save the summary
    """
    summary = {
        "total_comparisons": len(evaluation_results),
        "overall_winners": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0},
        "criteria_winners": {
            "Comprehensiveness": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0},
            "Diversity": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0},
            "Empowerment": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0}
        }
    }
    
    for result in evaluation_results:
        if "evaluation" in result and isinstance(result["evaluation"], dict):
            eval_data = result["evaluation"]

            if "Overall Winner" in eval_data and "Winner" in eval_data["Overall Winner"]:
                winner = eval_data["Overall Winner"]["Winner"]
                if winner in summary["overall_winners"]:
                    summary["overall_winners"][winner] += 1
                else:
                    summary["overall_winners"]["Unknown"] += 1
            else:
                summary["overall_winners"]["Unknown"] += 1
            
            # Count criteria winners
            for criterion in ["Comprehensiveness", "Diversity", "Empowerment"]:
                if criterion in eval_data and "Winner" in eval_data[criterion]:
                    winner = eval_data[criterion]["Winner"]
                    if winner in summary["criteria_winners"][criterion]:
                        summary["criteria_winners"][criterion][winner] += 1
                    else:
                        summary["criteria_winners"][criterion]["Unknown"] += 1
                else:
                    summary["criteria_winners"][criterion]["Unknown"] += 1
    
    # Calculate percentages
    valid_total = summary["overall_winners"]["Answer 1"] + summary["overall_winners"]["Answer 2"]
    if valid_total > 0:
        summary["percentages"] = {
            "Answer 1": round(summary["overall_winners"]["Answer 1"] / valid_total * 100, 2),
            "Answer 2": round(summary["overall_winners"]["Answer 2"] / valid_total * 100, 2)
        }
    
    # Save summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"Evaluation summary saved to {summary_file}")
    
    # Print quick results to console
    print("\n--- EVALUATION SUMMARY ---")
    print(f"Total comparisons: {summary['total_comparisons']}")
    print(f"Valid results: {valid_total}")
    if valid_total > 0:
        print(f"Answer 1 wins: {summary['overall_winners']['Answer 1']} ({summary['percentages']['Answer 1']}%)")
        print(f"Answer 2 wins: {summary['overall_winners']['Answer 2']} ({summary['percentages']['Answer 2']}%)")
    
    return summary

if __name__ == "__main__":

    query_file = "/Users/oraichain/Desktop/DATN/LightRAG/queries.txt"
    result1_file = "/Users/oraichain/Desktop/DATN/lightrag_mix_results.json"  
    result2_file = "/Users/oraichain/Desktop/DATN/hyde_results.json"       
    
    output_file = "/Users/oraichain/Desktop/DATN/direct_evaluation_results.json"
    
    # Run direct evaluation
    direct_eval(
        query_file=query_file,
        result1_file=result1_file,
        result2_file=result2_file,
        output_file=output_file
    )
import re
import json
import jsonlines
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

def batch_eval(query_file, result1_file, result2_file, output_file_path, batch_result_file=None):
    """
    Create batch evaluation requests comparing two sets of results.
    
    Args:
        query_file: File containing the queries
        result1_file: File containing the first set of results (Answer 1)
        result2_file: File containing the second set of results (Answer 2)
        output_file_path: File to write batch requests
        batch_result_file: File to save batch results (optional)
    """
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
    
    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

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

        request_data = {
            "custom_id": f"request-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        }

        requests.append(request_data)

    # Write requests to file
    with jsonlines.open(output_file_path, mode="w") as writer:
        for request in requests:
            writer.write(request)

    print(f"Batch API requests written to {output_file_path}")

    # Create batch
    batch_input_file = client.files.create(
        file=open(output_file_path, "rb"), purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "RAG Evaluation Comparison"},
    )

    print(f"Batch {batch.id} has been created.")
    
    batch_info = {
        "batch_id": batch.id,
        "input_file_id": batch_input_file_id,
        "query_file": query_file,
        "result1_file": result1_file,
        "result2_file": result2_file,
        "request_count": len(requests)
    }
    
    if batch_result_file:
        with open(batch_result_file, "w") as f:
            json.dump(batch_info, f, indent=4)
        print(f"Batch info saved to {batch_result_file}")
    
    return batch.id

def retrieve_batch_results(batch_id, output_file):
    """
    Retrieve and save results from a completed batch
    
    Args:
        batch_id: ID of the batch to retrieve
        output_file: File to save the results
    """
    client = OpenAI()
    
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        print(f"Batch status: {batch.status}. Not completed yet.")
        return False
    
    output_file_id = batch.output_file_id
    if not output_file_id:
        print("No output file available")
        return False
    
    # Download output file
    response = client.files.content(output_file_id)
    content = response.read().decode("utf-8")
    
    # Parse the jsonl content
    results = []
    for line in content.strip().split("\n"):
        results.append(json.loads(line))
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Batch results saved to {output_file}")
    return True

def summarize_evaluation(eval_results_file, summary_file):
    """
    Summarize the evaluation results
    
    Args:
        eval_results_file: File containing evaluation results
        summary_file: File to save the summary
    """
    with open(eval_results_file, "r") as f:
        results = json.load(f)
    
    summary = {
        "total_comparisons": len(results),
        "overall_winners": {"Answer 1": 0, "Answer 2": 0},
        "criteria_winners": {
            "Comprehensiveness": {"Answer 1": 0, "Answer 2": 0},
            "Diversity": {"Answer 1": 0, "Answer 2": 0},
            "Empowerment": {"Answer 1": 0, "Answer 2": 0}
        }
    }
    
    for result in results:
        if "body" in result and "choices" in result["body"]:
            # Extract JSON from the response
            try:
                content = result["body"]["choices"][0]["message"]["content"]
                eval_data = json.loads(content)
                
                # Count overall winners
                if "Overall Winner" in eval_data and "Winner" in eval_data["Overall Winner"]:
                    winner = eval_data["Overall Winner"]["Winner"]
                    if winner in summary["overall_winners"]:
                        summary["overall_winners"][winner] += 1
                
                # Count criteria winners
                for criterion in ["Comprehensiveness", "Diversity", "Empowerment"]:
                    if criterion in eval_data and "Winner" in eval_data[criterion]:
                        winner = eval_data[criterion]["Winner"]
                        if winner in summary["criteria_winners"][criterion]:
                            summary["criteria_winners"][criterion][winner] += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing result: {e}")
    
    # Calculate percentages
    total = summary["overall_winners"]["Answer 1"] + summary["overall_winners"]["Answer 2"]
    if total > 0:
        summary["percentages"] = {
            "Answer 1": round(summary["overall_winners"]["Answer 1"] / total * 100, 2),
            "Answer 2": round(summary["overall_winners"]["Answer 2"] / total * 100, 2)
        }
    
    # Save summary
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"Evaluation summary saved to {summary_file}")
    return summary

if __name__ == "__main__":
    # Example usage
    query_file = "/Users/oraichain/Desktop/DATN/LightRAG/queries.txt"
    result1_file = "/Users/oraichain/Desktop/DATN/lightrag_mix_results.json"  
    result2_file = "/Users/oraichain/Desktop/DATN/hyde_results.json"       
    
    # Files to create
    batch_requests_file = "/Users/oraichain/Desktop/DATN/eval_batch_requests_mix_hyde.jsonl"
    batch_info_file = "/Users/oraichain/Desktop/DATN/eval_batch_info_mix_hyde.json"
    
    batch_id = batch_eval(
        query_file=query_file,
        result1_file=result1_file,
        result2_file=result2_file,
        output_file_path=batch_requests_file,
        batch_result_file=batch_info_file
    )
    
    # print("\nAfter the batch completes, run:")
    # print(f"retrieve_batch_results('{batch_id}', 'eval_results.json')")
    # print("summarize_evaluation('eval_results.json', 'eval_summary.json')")
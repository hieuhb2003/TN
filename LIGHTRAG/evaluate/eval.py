import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score

# Thiết lập logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectRetrievalEvaluator:
    """
    Lớp đánh giá hiệu suất của các bộ query-document đã được retrieval sẵn,
    tính điểm trực tiếp từ thứ tự các tài liệu mà không cần embedding lại.
    """
    
    def __init__(self, show_progress_bar: bool = True):
        """
        Khởi tạo DirectRetrievalEvaluator.
        
        Args:
            show_progress_bar: Hiển thị thanh tiến trình khi tính toán
        """
        self.show_progress_bar = show_progress_bar
    
    def calculate_metrics(
        self,
        queries: List[str],
        retrieved_docs: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        at_k: int = 10
    ) -> Dict[str, float]:
        """
        Tính toán các chỉ số đánh giá trực tiếp từ thứ tự các tài liệu đã được retrieval.
        
        Args:
            queries: Danh sách các câu truy vấn
            retrieved_docs: Dictionary ánh xạ từ query đến danh sách các document được retrieval (theo thứ tự)
            ground_truth: Dictionary ánh xạ từ query đến danh sách các document liên quan (ground truth)
            at_k: Chỉ xem xét k document đầu tiên cho đánh giá
            
        Returns:
            Dictionary chứa các chỉ số đánh giá
        """
        all_ap_scores = []  # Average Precision
        all_mrr_scores = []  # Mean Reciprocal Rank
        all_ndcg_scores = []  # Normalized Discounted Cumulative Gain
        
        for query in queries:
            # Lấy các document được retrieval cho query này
            docs = retrieved_docs.get(query, [])
            
            # Lấy các document ground truth cho query này
            positive_docs = ground_truth.get(query, [])
            
            # Nếu không có document nào được retrieval hoặc không có ground truth, bỏ qua query này
            if not docs or not positive_docs:
                logger.warning(f"Bỏ qua query không có document hoặc ground truth: {query}")
                continue
            
            # Giới hạn số lượng document xem xét
            docs = docs[:at_k]
            
            # Tạo nhãn relevance (1 nếu document nằm trong ground truth, 0 nếu không)
            relevance = [1 if doc in positive_docs else 0 for doc in docs]
            
            # Tính MRR (Mean Reciprocal Rank)
            mrr = 0
            for i, rel in enumerate(relevance):
                if rel == 1:
                    mrr = 1.0 / (i + 1)
                    break
            all_mrr_scores.append(mrr)
            
            # Tính NDCG (Normalized Discounted Cumulative Gain)
            # Tạo scores giả định (1.0 cho tất cả các document để giữ nguyên thứ tự)
            scores = [1.0] * len(docs)
            
            # Tính NDCG
            ndcg = ndcg_score([relevance], [scores], k=at_k)
            all_ndcg_scores.append(ndcg)
            
            # Tính AP (Average Precision)
            # Cần tạo scores cho tất cả các document (cả positive và negative)
            # Giả định scores giảm dần theo thứ tự
            all_docs = docs + [doc for doc in positive_docs if doc not in docs]
            all_relevance = [1 if doc in positive_docs else 0 for doc in all_docs]
            all_scores = [1.0 / (i + 1) for i in range(len(all_docs))]  # Scores giảm dần
            
            ap = average_precision_score(all_relevance, all_scores)
            all_ap_scores.append(ap)
        
        # Tính trung bình
        mean_ap = np.mean(all_ap_scores) if all_ap_scores else 0
        mean_mrr = np.mean(all_mrr_scores) if all_mrr_scores else 0
        mean_ndcg = np.mean(all_ndcg_scores) if all_ndcg_scores else 0
        
        metrics = {
            "map": mean_ap,
            f"mrr@{at_k}": mean_mrr,
            f"ndcg@{at_k}": mean_ndcg
        }
        
        return metrics
    
    def evaluate_model(
        self,
        queries: List[str],
        retrieved_docs: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        at_k: int = 10,
        name: str = "",
        output_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Đánh giá hiệu suất retrieval của một mô hình.
        
        Args:
            queries: Danh sách các câu truy vấn
            retrieved_docs: Dictionary ánh xạ từ query đến danh sách các document được retrieval (theo thứ tự)
            ground_truth: Dictionary ánh xạ từ query đến danh sách các document liên quan (ground truth)
            at_k: Chỉ xem xét k document đầu tiên cho đánh giá
            name: Tên của mô hình
            output_path: Đường dẫn để lưu kết quả đánh giá
            
        Returns:
            Dictionary chứa các chỉ số đánh giá
        """
        # Tạo thư mục output nếu cần
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Tính toán các chỉ số đánh giá
        logger.info(f"Đánh giá {name} với {len(queries)} queries, at_k={at_k}")
        metrics = self.calculate_metrics(queries, retrieved_docs, ground_truth, at_k)
        
        # In kết quả
        logger.info(f"Kết quả đánh giá {name}:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Lưu kết quả vào file CSV nếu cần
        if output_path:
            csv_path = os.path.join(output_path, f"{name}_results_@{at_k}.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("metric,value\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"{metric_name},{metric_value:.6f}\n")
        
        return metrics
    
    def evaluate_multiple_models(
        self,
        queries: List[str],
        model_results: Dict[str, Dict[str, List[str]]],
        ground_truth: Dict[str, List[str]],
        at_k: int = 10,
        output_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Đánh giá hiệu suất của nhiều mô hình retrieval.
        
        Args:
            queries: Danh sách các câu truy vấn
            model_results: Dictionary ánh xạ từ tên mô hình đến kết quả retrieval của mô hình đó
                           (mỗi kết quả là một dictionary ánh xạ từ query đến danh sách các document theo thứ tự)
            ground_truth: Dictionary ánh xạ từ query đến danh sách các document liên quan (ground truth)
            at_k: Chỉ xem xét k document đầu tiên cho đánh giá
            output_path: Đường dẫn để lưu kết quả đánh giá
            
        Returns:
            Dictionary ánh xạ từ tên mô hình đến các chỉ số đánh giá của mô hình đó
        """
        # Tạo thư mục output nếu cần
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
        
        results = {}
        
        for model_name, retrieved_docs in model_results.items():
            logger.info(f"\nĐánh giá mô hình: {model_name}")
            
            # Đánh giá
            metrics = self.evaluate_model(
                queries=queries,
                retrieved_docs=retrieved_docs,
                ground_truth=ground_truth,
                at_k=at_k,
                name=model_name,
                output_path=output_path
            )
            
            results[model_name] = metrics
        
        # Lưu kết quả tổng hợp
        if output_path:
            with open(os.path.join(output_path, "evaluation_summary.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def load_data_from_json(self, file_path: str) -> Dict[str, Any]:
        """
        Tải dữ liệu từ file JSON.
        
        Args:
            file_path: Đường dẫn đến file JSON
            
        Returns:
            Dữ liệu được tải từ file JSON
        """
        logger.info(f"Tải dữ liệu từ: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def save_data_to_json(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Lưu dữ liệu vào file JSON.
        
        Args:
            data: Dữ liệu cần lưu
            file_path: Đường dẫn đến file JSON
        """
        logger.info(f"Lưu dữ liệu vào: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo evaluator
    evaluator = DirectRetrievalEvaluator()
    
    # Ví dụ dữ liệu
    queries = [
        "What is LightRAG?",
        "How does BGE-M3 work?",
        "What are the benefits of knowledge graphs?"
    ]
    
    # Kết quả retrieval từ các mô hình khác nhau (theo thứ tự giảm dần về độ liên quan)
    model_results = {
        "LightRAG_hybrid": {
            "What is LightRAG?": [
                "LightRAG is a simple and fast Retrieval-Augmented Generation system.",
                "LightRAG uses a hybrid approach combining knowledge graphs and vector search.",
                "RAG systems retrieve relevant documents and then generate responses based on them."
            ],
            "How does BGE-M3 work?": [
                "BGE-M3 is a powerful embedding model for text retrieval.",
                "BGE-M3 is based on the BERT architecture with improvements for multilingual support.",
                "Vector databases store embeddings for efficient similarity search."
            ],
            "What are the benefits of knowledge graphs?": [
                "Knowledge graphs can improve retrieval by capturing relationships between entities.",
                "Knowledge graphs represent relationships between entities as a graph structure.",
                "LightRAG uses a hybrid approach combining knowledge graphs and vector search."
            ]
        },
        "Naive_RAG": {
            "What is LightRAG?": [
                "LightRAG is a simple and fast Retrieval-Augmented Generation system.",
                "RAG systems retrieve relevant documents and then generate responses based on them.",
                "Retrieval-Augmented Generation combines retrieval and generation for better results."
            ],
            "How does BGE-M3 work?": [
                "BGE-M3 is a powerful embedding model for text retrieval.",
                "Vector databases store embeddings for efficient similarity search.",
                "Retrieval-Augmented Generation combines retrieval and generation for better results."
            ],
            "What are the benefits of knowledge graphs?": [
                "Knowledge graphs can improve retrieval by capturing relationships between entities.",
                "Vector databases store embeddings for efficient similarity search.",
                "LightRAG uses a hybrid approach combining knowledge graphs and vector search."
            ]
        }
    }
    
    # Ground truth
    ground_truth = {
        "What is LightRAG?": [
            "LightRAG is a simple and fast Retrieval-Augmented Generation system.",
            "LightRAG uses a hybrid approach combining knowledge graphs and vector search."
        ],
        "How does BGE-M3 work?": [
            "BGE-M3 is a powerful embedding model for text retrieval.",
            "BGE-M3 is based on the BERT architecture with improvements for multilingual support."
        ],
        "What are the benefits of knowledge graphs?": [
            "Knowledge graphs can improve retrieval by capturing relationships between entities.",
            "Knowledge graphs represent relationships between entities as a graph structure."
        ]
    }
    
    # Đánh giá các mô hình
    results = evaluator.evaluate_multiple_models(
        queries=queries,
        model_results=model_results,
        ground_truth=ground_truth,
        at_k=3,
        output_path="./direct_evaluation_results"
    )
    
    # In kết quả tổng hợp
    print("\nKết quả tổng hợp:")
    for model_name, metrics in results.items():
        print(f"\nMô hình: {model_name}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Ví dụ về cách tải dữ liệu từ file JSON
    print("\nVí dụ về cách tải dữ liệu từ file JSON:")
    print("# Lưu dữ liệu mẫu vào file JSON")
    evaluator.save_data_to_json(model_results, "model_results.json")
    evaluator.save_data_to_json(ground_truth, "ground_truth.json")
    
    print("# Tải dữ liệu từ file JSON")
    loaded_model_results = evaluator.load_data_from_json("model_results.json")
    loaded_ground_truth = evaluator.load_data_from_json("ground_truth.json")
    
    print("# Đánh giá với dữ liệu đã tải")
    loaded_results = evaluator.evaluate_multiple_models(
        queries=queries,
        model_results=loaded_model_results,
        ground_truth=loaded_ground_truth,
        at_k=3,
        output_path="./direct_evaluation_results_loaded"
    ) 
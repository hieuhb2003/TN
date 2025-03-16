import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.util import cos_sim

# Thiết lập logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """
    Lớp đánh giá hiệu suất của các bộ query-document đã được retrieval sẵn.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        device: Optional[str] = None
    ):
        """
        Khởi tạo RetrievalEvaluator.
        
        Args:
            model_name: Tên của mô hình embedding sử dụng để đánh giá
            batch_size: Kích thước batch khi tính embedding
            show_progress_bar: Hiển thị thanh tiến trình khi tính toán
            device: Thiết bị để chạy mô hình (None để tự động chọn)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        
        logger.info(f"Khởi tạo mô hình embedding: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
    
    def prepare_samples(
        self,
        queries: List[str],
        retrieved_docs: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Chuẩn bị dữ liệu mẫu cho RerankingEvaluator.
        
        Args:
            queries: Danh sách các câu truy vấn
            retrieved_docs: Dictionary ánh xạ từ query đến danh sách các document được retrieval
            ground_truth: Dictionary ánh xạ từ query đến danh sách các document liên quan (ground truth)
            
        Returns:
            Danh sách các mẫu theo định dạng của RerankingEvaluator
        """
        samples = []
        
        for query in queries:
            # Lấy các document được retrieval cho query này
            docs = retrieved_docs.get(query, [])
            
            # Lấy các document ground truth cho query này
            positive_docs = ground_truth.get(query, [])
            
            # Nếu không có document nào được retrieval hoặc không có ground truth, bỏ qua query này
            if not docs or not positive_docs:
                logger.warning(f"Bỏ qua query không có document hoặc ground truth: {query}")
                continue
            
            # Tất cả các document được retrieval mà không nằm trong ground truth được coi là negative
            negative_docs = [doc for doc in docs if doc not in positive_docs]
            
            # Tạo mẫu
            sample = {
                "query": query,
                "positive": positive_docs,
                "negative": negative_docs
            }
            
            samples.append(sample)
        
        logger.info(f"Đã chuẩn bị {len(samples)} mẫu cho đánh giá")
        return samples
    
    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        at_k: int = 10,
        name: str = "",
        output_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Đánh giá hiệu suất retrieval sử dụng RerankingEvaluator.
        
        Args:
            samples: Danh sách các mẫu theo định dạng của RerankingEvaluator
            at_k: Chỉ xem xét k document đầu tiên cho đánh giá
            name: Tên của đánh giá
            output_path: Đường dẫn để lưu kết quả đánh giá
            
        Returns:
            Dictionary chứa các chỉ số đánh giá
        """
        # Tạo thư mục output nếu cần
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Khởi tạo RerankingEvaluator
        evaluator = RerankingEvaluator(
            samples=samples,
            at_k=at_k,
            name=name,
            write_csv=True if output_path else False,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size
        )
        
        # Đánh giá
        logger.info(f"Đánh giá {name} với {len(samples)} mẫu, at_k={at_k}")
        metrics = evaluator(self.model, output_path=output_path)
        
        # In kết quả
        logger.info(f"Kết quả đánh giá {name}:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
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
                           (mỗi kết quả là một dictionary ánh xạ từ query đến danh sách các document)
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
            
            # Chuẩn bị mẫu cho mô hình này
            samples = self.prepare_samples(queries, retrieved_docs, ground_truth)
            
            # Đánh giá
            metrics = self.evaluate(
                samples=samples,
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
    evaluator = RetrievalEvaluator(model_name="BAAI/bge-m3")
    
    # Ví dụ dữ liệu
    queries = [
        "What is LightRAG?",
        "How does BGE-M3 work?",
        "What are the benefits of knowledge graphs?"
    ]
    
    # Kết quả retrieval từ các mô hình khác nhau
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
        output_path="./evaluation_results"
    )
    
    # In kết quả tổng hợp
    print("\nKết quả tổng hợp:")
    for model_name, metrics in results.items():
        print(f"\nMô hình: {model_name}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Ví dụ về cách tải dữ liệu từ file JSON
    # Giả sử bạn đã lưu dữ liệu vào các file JSON
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
        output_path="./evaluation_results_loaded"
    ) 
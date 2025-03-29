import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Thiết lập logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_recall(retrieved_docs, relevant_docs, k=None):
    """
    Tính Recall dựa trên danh sách tài liệu được truy vấn và tài liệu liên quan.
    
    Args:
        retrieved_docs: Danh sách các tài liệu được truy vấn (theo thứ tự)
        relevant_docs: Danh sách các tài liệu liên quan (ground truth)
        k: Giới hạn số lượng tài liệu xem xét (default: None - xem xét tất cả)
    
    Returns:
        Điểm Recall
    """
    if k is not None:
        retrieved_docs = retrieved_docs[:k]
    
    # Nếu không có tài liệu liên quan, trả về 0 để tránh chia cho 0
    if not relevant_docs:
        return 0.0
    
    # Tạo set để tìm kiếm nhanh hơn
    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_docs)
    
    # Đếm số lượng tài liệu liên quan được tìm thấy trong kết quả truy vấn
    hits = len(relevant_set.intersection(retrieved_set))
    
    # Tính recall: số lượng tài liệu liên quan được tìm thấy / tổng số tài liệu liên quan
    recall = hits / len(relevant_set)
    return recall


def calculate_mrr(retrieved_docs, relevant_docs, k=None):
    """
    Tính Mean Reciprocal Rank (MRR) dựa trên danh sách tài liệu được truy vấn và tài liệu liên quan.
    
    Args:
        retrieved_docs: Danh sách các tài liệu được truy vấn (theo thứ tự)
        relevant_docs: Danh sách các tài liệu liên quan (ground truth)
        k: Giới hạn số lượng tài liệu xem xét (default: None - xem xét tất cả)
    
    Returns:
        Điểm MRR
    """
    if k is not None:
        retrieved_docs = retrieved_docs[:k]
    
    # Nếu không có tài liệu liên quan hoặc không có tài liệu truy vấn, trả về 0
    if not retrieved_docs or not relevant_docs:
        return 0.0
    
    # Tạo set tài liệu liên quan để tìm kiếm nhanh hơn
    relevant_set = set(relevant_docs)
    
    # Tìm vị trí đầu tiên của bất kỳ tài liệu liên quan nào trong danh sách truy vấn
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_set:
            # MRR = 1 / (vị trí đầu tiên của relevant item)
            return 1.0 / (i + 1)
    
    # Nếu không tìm thấy tài liệu liên quan nào trong danh sách truy vấn
    return 0.0



class RecallMRREvaluator:
    """
    Lớp đánh giá hiệu suất của các bộ query-document chỉ tính Recall và MRR@10
    """
    
    def __init__(self, show_progress_bar: bool = True):
        """
        Khởi tạo RecallMRREvaluator.
        
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
        Tính toán Recall và MRR từ thứ tự các tài liệu đã được retrieval.
        
        Args:
            queries: Danh sách các câu truy vấn
            retrieved_docs: Dictionary ánh xạ từ query đến danh sách các document được retrieval (theo thứ tự)
            ground_truth: Dictionary ánh xạ từ query đến danh sách các document liên quan (ground truth)
            at_k: Chỉ xem xét k document đầu tiên cho đánh giá
            
        Returns:
            Dictionary chứa các chỉ số đánh giá
        """
        all_recall_scores = []
        all_mrr_scores = []
        
        processed_queries = 0
        skipped_queries = 0
        
        for query in queries:
            # Kiểm tra xem query có tồn tại trong retrieved_docs và ground_truth không
            if query not in retrieved_docs or query not in ground_truth:
                skipped_queries += 1
                logger.warning(f"Query không có trong retrieved_docs hoặc ground_truth: {query[:50]}...")
                continue
            
            # Lấy các document được retrieval cho query này
            docs = retrieved_docs[query]
            
            # Lấy các document ground truth cho query này
            positive_docs = ground_truth[query]
            
            # Nếu không có document nào được retrieval hoặc không có ground truth, bỏ qua query này
            if not docs or not positive_docs:
                skipped_queries += 1
                logger.warning(f"Bỏ qua query không có document hoặc ground truth: {query[:50]}...")
                continue
            
            # Giới hạn số lượng document xem xét cho đánh giá
            docs_for_eval = docs[:at_k] if at_k is not None else docs
            
            # Tính Recall
            recall = calculate_recall(docs_for_eval, positive_docs)
            all_recall_scores.append(recall)
            
            # Tính MRR (Mean Reciprocal Rank)
            mrr = calculate_mrr(docs_for_eval, positive_docs)
            all_mrr_scores.append(mrr)
            
            processed_queries += 1
        
        logger.info(f"Đã xử lý {processed_queries} queries, bỏ qua {skipped_queries} queries")
        
        # Tính trung bình
        mean_recall = np.mean(all_recall_scores) if all_recall_scores else 0
        mean_mrr = np.mean(all_mrr_scores) if all_mrr_scores else 0
        
        metrics = {
            f"recall@{at_k}": mean_recall,
            f"mrr@{at_k}": mean_mrr,
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
        try:
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
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá mô hình {name}: {str(e)}")
            return {"error": str(e)}
    
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
        
        # Kiểm tra cấu trúc model_results
        if not isinstance(model_results, dict):
            raise ValueError("model_results phải là một dictionary")
            
        for model_name, retrieved_docs in model_results.items():
            logger.info(f"\nĐánh giá mô hình: {model_name}")
            
            # Kiểm tra cấu trúc retrieved_docs
            if not isinstance(retrieved_docs, dict):
                logger.error(f"Dữ liệu của mô hình {model_name} không đúng định dạng. Cần là dictionary.")
                continue
                
            # Đánh giá
            try:
                metrics = self.evaluate_model(
                    queries=queries,
                    retrieved_docs=retrieved_docs,
                    ground_truth=ground_truth,
                    at_k=at_k,
                    name=model_name,
                    output_path=output_path
                )
                
                results[model_name] = metrics
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá mô hình {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        if output_path:
            try:
                with open(os.path.join(output_path, "evaluation_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Lỗi khi lưu kết quả đánh giá: {str(e)}")
        
        return results
    
    def load_data_from_json(self, file_path: str) -> Dict[str, Any]:
        """
        Tải dữ liệu từ file JSON.
        
        Args:
            file_path: Đường dẫn đến file JSON
            
        Returns:
            Dictionary chứa dữ liệu từ file JSON
        """
        logger.info(f"Tải dữ liệu từ: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu từ {file_path}: {str(e)}")
            raise
    
    def save_data_to_json(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Lưu dữ liệu vào file JSON.
        
        Args:
            data: Dữ liệu cần lưu
            file_path: Đường dẫn đến file JSON
        """
        logger.info(f"Lưu dữ liệu vào: {file_path}")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu vào {file_path}: {str(e)}")
            raise

    def evaluate_from_result_file(
        self,
        result_file: str,
        ground_truth_file: str,
        at_k: int = 10,
        output_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Đánh giá từ file kết quả và ground truth.
        
        Args:
            result_file: Đường dẫn đến file kết quả theo định dạng 
                         {model_name -> {query -> [documents]}}
            ground_truth_file: Đường dẫn đến file ground truth theo định dạng
                               {query -> [relevant documents]}
            at_k: Chỉ xem xét k document đầu tiên cho đánh giá
            output_path: Đường dẫn để lưu kết quả đánh giá
            
        Returns:
            Dictionary ánh xạ từ tên mô hình đến các chỉ số đánh giá của mô hình đó
        """
        # Tải dữ liệu
        model_results = self.load_data_from_json(result_file)
        ground_truth = self.load_data_from_json(ground_truth_file)
        
        # Lấy danh sách các câu truy vấn từ ground truth
        queries = list(ground_truth.keys())
        
        # Đánh giá
        return self.evaluate_multiple_models(
            queries=queries,
            model_results=model_results,
            ground_truth=ground_truth,
            at_k=at_k,
            output_path=output_path
        )


# Hàm để xử lý file kết quả cho việc đánh giá
def process_result_file_for_evaluation(
    result_file: str,
    output_file: str = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    Chuyển đổi cấu trúc dữ liệu từ result.json sang dạng phù hợp cho việc đánh giá
    
    Args:
        result_file: Đường dẫn đến file kết quả
        output_file: Đường dẫn để lưu cấu trúc mới (nếu cần)
        
    Returns:
        Dữ liệu đã được chuyển đổi
    """
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Cấu trúc kết quả: {model: {query: [documents]}}
        processed_results = {}
        
        for model_name, model_data in results.items():
            if not isinstance(model_data, dict):
                logger.warning(f"Dữ liệu của mô hình {model_name} không đúng định dạng. Bỏ qua.")
                continue
                
            model_results = {}
            for query, docs in model_data.items():
                if isinstance(docs, list):
                    model_results[query] = docs
                else:
                    logger.warning(f"Kết quả của query '{query[:30]}...' trong mô hình {model_name} không phải list. Bỏ qua.")
            
            processed_results[model_name] = model_results
        
        # Lưu kết quả đã xử lý nếu cần
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_results, f, indent=2, ensure_ascii=False)
        
        return processed_results
    
    except Exception as e:
        logger.error(f"Lỗi khi xử lý file kết quả: {str(e)}")
        raise


# Ví dụ sử dụng
if __name__ == "__main__":
    try:
        evaluator = RecallMRREvaluator()
        
        # Đường dẫn đến các file
        result_file = "/home/hungpv/projects/TN/LIGHTRAG/result_26_3/retrieval_results_true_method.json"
        # output_processed_file = "processed_result.json"
        ground_truth_file = "/home/hungpv/projects/TN/LIGHTRAG/data/small_grouth_truth.json"
        
        # Tiền xử lý dữ liệu nếu cần
        # processed_results = process_result_file_for_evaluation(
        #     result_file=result_file,
        #     output_file=output_processed_file
        # )
        with open(result_file, 'r', encoding='utf-8') as f:
            processed_results = json.load(f)        
        # Tải ground truth 
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
        except FileNotFoundError:
            logger.warning(f"File ground truth {ground_truth_file} không tồn tại. Tạo một ground truth tạm thời cho demo.")
            # Tạo ground truth tạm thời nếu cần
            # ...
        
        # Lấy danh sách các câu truy vấn từ ground truth
        queries = list(ground_truth.keys())
        
        # Đánh giá các mô hình
        results = evaluator.evaluate_multiple_models(
            queries=queries,
            model_results=processed_results,
            ground_truth=ground_truth,
            at_k=10,
            output_path="./evaluation_results_mrr_recall_new_true_method"
        )
        
        # In kết quả tổng hợp
        print("\nKết quả tổng hợp:")
        for model_name, metrics in results.items():
            print(f"\nMô hình: {model_name}")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
    
    except Exception as e:
        logger.error(f"Lỗi chính: {str(e)}")
        import traceback
        traceback.print_exc()
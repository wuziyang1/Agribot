"""
稠密向量检索性能评估脚本

评估指标：
- Recall@1: 第一个结果是否正确
- Recall@3: 前3个结果中是否包含正确文档
- MRR: 平均倒数排名
"""

import json
import logging
import sys
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import MilvusClient
from agribot_chat.rag.rag_config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    recall_at_1: float
    recall_at_3: float
    precision_at_1: float
    precision_at_3: float
    mrr: float
    total_queries: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Recall@1": f"{self.recall_at_1:.4f}",
            "Recall@3": f"{self.recall_at_3:.4f}",
            "Precision@1": f"{self.precision_at_1:.4f}",
            "Precision@3": f"{self.precision_at_3:.4f}",
            "MRR": f"{self.mrr:.4f}",
            "总查询数": self.total_queries
        }


class DenseEvaluator:
    """稠密向量检索评估器"""
    
    def __init__(self):
        self.milvus_client = None
        self.embeddings = None
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化组件"""
        try:
            self._initialize_embeddings()
            self._initialize_milvus()
            logger.info("✅ 稠密向量评估器初始化完成")
        except Exception as e:
            logger.error(f"❌ 评估器初始化失败: {e}")
            raise
    
    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        from openai import OpenAI
        
        class CustomEmbeddings:
            def __init__(self, model_name: str, api_key: str, api_base: str, dimensions: int):
                self.model_name = model_name
                self.client = OpenAI(api_key=api_key, base_url=api_base)
                self.dimensions = dimensions
            
            def embed_query(self, text: str) -> List[float]:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=[text],
                    dimensions=self.dimensions,
                    encoding_format="float"
                )
                return response.data[0].embedding
        
        self.embeddings = CustomEmbeddings(
            model_name=Config.LLM_EMBEDDING_MODEL_NAME,
            api_key=Config.LLM_EMBEDDING_API_KEY,
            api_base=Config.LLM_EMBEDDING_BASE_URL,
            dimensions=Config.MILVUS_VECTOR_DIM
        )
        logger.info(f"✅ 嵌入模型初始化成功: {Config.LLM_EMBEDDING_MODEL_NAME}")
    
    def _initialize_milvus(self):
        """初始化 Milvus 客户端"""
        uri = f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}"
        self.milvus_client = MilvusClient(
            uri=uri,
            db_name=Config.MILVUS_DATABASE or "default",
            user=Config.MILVUS_USER or "",
            password=Config.MILVUS_PASSWORD or "",
        )
        logger.info(f"✅ Milvus 客户端初始化成功: {uri}")
    
    def dense_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """稠密向量检索"""
        try:
            query_vector = self.embeddings.embed_query(query)
            
            results = self.milvus_client.search(
                collection_name=Config.MILVUS_COLLECTION_NAME,
                data=[query_vector],
                anns_field="content_vector",
                search_params={
                    "metric_type": "COSINE",
                    "params": {"nprobe": 64}
                },
                limit=top_k,
                output_fields=["doc_name", "doc_path_name", "doc_type", "content"],
            )
            
            formatted_results = []
            if results and results[0]:
                for hit in results[0]:
                    entity = hit.get("entity", {})
                    formatted_results.append({
                        "doc_name": entity.get("doc_name", ""),
                        "doc_path_name": entity.get("doc_path_name", ""),
                        "doc_type": entity.get("doc_type", ""),
                        "content": entity.get("content", ""),
                        "score": float(hit.get("distance", 0.0))
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"❌ 向量检索失败: {e}")
            return []
    
    def calculate_metrics(self, test_data: List[Dict[str, Any]], top_k: int = 3) -> EvaluationMetrics:
        """计算评估指标"""
        recall_at_1_count = 0
        recall_at_3_count = 0
        precision_at_1_sum = 0.0
        precision_at_3_sum = 0.0
        mrr_sum = 0.0
        total = len(test_data)
        
        for i, item in enumerate(test_data, 1):
            question = item["question"]
            source_pdf = item["source_pdf"]
            
            results = self.dense_search(question, top_k=top_k)
            
            if not results:
                logger.warning(f"⚠️ 查询 {i}/{total} 无检索结果")
                continue
            
            rank = None
            for idx, result in enumerate(results, 1):
                doc_path = result.get("doc_path_name", "")
                if source_pdf in doc_path:
                    rank = idx
                    break
            
            if rank is not None:
                if rank == 1:
                    recall_at_1_count += 1
                    precision_at_1_sum += 1.0
                
                if rank <= 3:
                    recall_at_3_count += 1
                    precision_at_3_sum += 1.0 / 3.0
                
                mrr_sum += 1.0 / rank
            
            if i % 10 == 0:
                logger.info(f"📊 评估进度: {i}/{total}")
        
        return EvaluationMetrics(
            recall_at_1=recall_at_1_count / total,
            recall_at_3=recall_at_3_count / total,
            precision_at_1=precision_at_1_sum / total,
            precision_at_3=precision_at_3_sum / total,
            mrr=mrr_sum / total,
            total_queries=total
        )
    
    def evaluate(self, test_data_path: str, output_path: str):
        """执行评估"""
        logger.info(f"📂 加载测试数据: {test_data_path}")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        logger.info(f"✅ 加载了 {len(test_data)} 条测试数据")
        
        logger.info("\n" + "="*60)
        logger.info("📊 开始稠密向量检索评估")
        logger.info("="*60)
        
        metrics = self.calculate_metrics(test_data, top_k=3)
        
        logger.info("\n" + "="*60)
        logger.info("📊 稠密向量检索评估结果")
        logger.info("="*60)
        logger.info(f"Recall@1:    {metrics.recall_at_1:.4f}")
        logger.info(f"Recall@3:    {metrics.recall_at_3:.4f}")
        logger.info(f"Precision@1: {metrics.precision_at_1:.4f}")
        logger.info(f"Precision@3: {metrics.precision_at_3:.4f}")
        logger.info(f"MRR:         {metrics.mrr:.4f}")
        logger.info("="*60)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"Dense": metrics.to_dict()}, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 结果已保存到: {output_path}")


def main():
    test_data_path = "experiment/generate_data/gen_data.json"
    output_path = "experiment/dense_evaluation_results.json"
    
    evaluator = DenseEvaluator()
    evaluator.evaluate(test_data_path, output_path)


if __name__ == "__main__":
    main()

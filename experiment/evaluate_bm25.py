"""
BM25 检索性能评估脚本

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
logging.getLogger("jieba").setLevel(logging.WARNING)


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


class BM25Evaluator:
    """BM25 检索评估器"""
    
    def __init__(self):
        self.milvus_client = None
        self.bm25_index = None
        self.bm25_corpus = []
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化组件"""
        try:
            self._initialize_milvus()
            self._check_bm25_support()
            self._initialize_bm25_index()
            logger.info("✅ BM25 评估器初始化完成")
        except Exception as e:
            logger.error(f"❌ 评估器初始化失败: {e}")
            raise
    
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
    
    def _check_bm25_support(self):
        """检测 BM25 支持"""
        try:
            import rank_bm25
            import jieba
            logger.info("✅ BM25 支持已启用")
        except ImportError:
            logger.error("❌ 缺少依赖库，请安装: pip install rank-bm25 jieba")
            raise
    
    def _initialize_bm25_index(self):
        """初始化 BM25 索引"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            
            logger.info("="*60)
            logger.info("📚 初始化 BM25 索引...")
            logger.info("="*60)
            
            # 从 Milvus 加载文档
            try:
                results = self.milvus_client.query(
                    collection_name=Config.MILVUS_COLLECTION_NAME,
                    filter="",
                    output_fields=["doc_name", "doc_path_name", "doc_type", "content"],
                    limit=10000
                )
                self.bm25_corpus = results
                logger.info(f"✅ 加载了 {len(results)} 个文档")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载所有文档: {e}")
                logger.info("💡 尝试使用采样方式...")
                
                # 采样方式
                from openai import OpenAI
                
                class CustomEmbeddings:
                    def __init__(self):
                        self.client = OpenAI(
                            api_key=Config.LLM_EMBEDDING_API_KEY,
                            base_url=Config.LLM_EMBEDDING_BASE_URL
                        )
                    
                    def embed_query(self, text: str) -> List[float]:
                        response = self.client.embeddings.create(
                            model=Config.LLM_EMBEDDING_MODEL_NAME,
                            input=[text],
                            dimensions=Config.MILVUS_VECTOR_DIM,
                            encoding_format="float"
                        )
                        return response.data[0].embedding
                
                embeddings = CustomEmbeddings()
                sample_queries = ["种植", "栽培", "技术", "方法", "病虫害"]
                self.bm25_corpus = []
                
                for sample_query in sample_queries:
                    sample_vector = embeddings.embed_query(sample_query)
                    sample_results = self.milvus_client.search(
                        collection_name=Config.MILVUS_COLLECTION_NAME,
                        data=[sample_vector],
                        anns_field="content_vector",
                        search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
                        limit=200,
                        output_fields=["doc_name", "doc_path_name", "doc_type", "content"],
                    )
                    
                    if sample_results and sample_results[0]:
                        for hit in sample_results[0]:
                            entity = hit.get("entity", {})
                            self.bm25_corpus.append(entity)
                
                # 去重
                seen = set()
                unique_corpus = []
                for doc in self.bm25_corpus:
                    doc_id = doc.get("doc_path_name", "")
                    if doc_id and doc_id not in seen:
                        seen.add(doc_id)
                        unique_corpus.append(doc)
                
                self.bm25_corpus = unique_corpus
                logger.info(f"✅ 采样了 {len(self.bm25_corpus)} 个文档")
            
            # 分词
            logger.info("🔧 对文档进行分词...")
            tokenized_corpus = []
            for doc in self.bm25_corpus:
                content = doc.get("content", "")
                tokens = list(jieba.cut(content))
                tokenized_corpus.append(tokens)
            
            # 创建 BM25 索引
            logger.info("🔧 创建 BM25 索引...")
            self.bm25_index = BM25Okapi(tokenized_corpus)
            logger.info("✅ BM25 索引创建完成")
            
        except Exception as e:
            logger.error(f"❌ BM25 索引初始化失败: {e}")
            raise
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 检索"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            
            query_tokens = list(jieba.cut(query))
            scores = self.bm25_index.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.bm25_corpus):
                    doc = self.bm25_corpus[idx]
                    results.append({
                        "doc_name": doc.get("doc_name", ""),
                        "doc_path_name": doc.get("doc_path_name", ""),
                        "doc_type": doc.get("doc_type", ""),
                        "content": doc.get("content", ""),
                        "score": float(scores[idx])
                    })
            
            return results
        except Exception as e:
            logger.error(f"❌ BM25 检索失败: {e}")
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
            
            results = self.bm25_search(question, top_k=top_k)
            
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
        logger.info("📊 开始 BM25 检索评估")
        logger.info("="*60)
        
        metrics = self.calculate_metrics(test_data, top_k=3)
        
        logger.info("\n" + "="*60)
        logger.info("📊 BM25 检索评估结果")
        logger.info("="*60)
        logger.info(f"Recall@1:    {metrics.recall_at_1:.4f}")
        logger.info(f"Recall@3:    {metrics.recall_at_3:.4f}")
        logger.info(f"Precision@1: {metrics.precision_at_1:.4f}")
        logger.info(f"Precision@3: {metrics.precision_at_3:.4f}")
        logger.info(f"MRR:         {metrics.mrr:.4f}")
        logger.info("="*60)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"BM25": metrics.to_dict()}, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 结果已保存到: {output_path}")


def main():
    test_data_path = "experiment/generate_data/gen_data.json"
    output_path = "experiment/bm25_evaluation_results.json"
    
    evaluator = BM25Evaluator()
    evaluator.evaluate(test_data_path, output_path)


if __name__ == "__main__":
    main()

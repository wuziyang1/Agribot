"""
重排序服务模块

支持多个重排序服务提供商：
- 阿里百炼平台 (dashscope)
- 硅基流动平台 (siliconflow)
"""
import logging
import requests
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from mildoc_chat.rag_config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankProvider(Enum):
    DASHSCOPE = "dashscope"
    SILICONFLOW = "siliconflow"


class RerankDocument(BaseModel):
    index: int
    content: str
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None


class RerankResponse(BaseModel):
    documents: List[RerankDocument]
    success: bool = True
    error_message: Optional[str] = None


class RerankService:
    def __init__(self, provider: RerankProvider, api_key: str, model_name: str, endpoint: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.endpoint = endpoint
        logger.info(f"重排序服务初始化完成: provider={provider.value}, model={model_name}, endpoint={self.endpoint}")

    def rerank_documents(self, query: str, documents: List[str], top_n: Optional[int] = None) -> RerankResponse:
        try:
            if not query or not documents:
                return RerankResponse(documents=[], success=False, error_message="查询或文档列表为空")
            if self.provider == RerankProvider.DASHSCOPE:
                response = self._rerank_dashscope(query, documents, top_n)
            elif self.provider == RerankProvider.SILICONFLOW:
                response = self._rerank_siliconflow(query, documents, top_n)
            else:
                raise ValueError(f"不支持的重排序提供商: {self.provider}")
            return response
        except Exception as e:
            logger.error(f"❌ 重排序失败: {e}")
            return RerankResponse(documents=[], success=False, error_message=str(e))

    def _rerank_dashscope(self, query: str, documents: List[str], top_n: Optional[int] = None) -> RerankResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": True, "top_n": top_n or len(documents)}
        }
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        rerank_docs = []
        if "output" in result and "results" in result["output"]:
            for item in result["output"]["results"]:
                document = item.get("document", {})
                content = document.get("text", "") if isinstance(document, dict) else str(document)
                rerank_docs.append(RerankDocument(
                    index=item.get("index", 0), content=content,
                    relevance_score=float(item.get("relevance_score", 0.0))
                ))
        else:
            return RerankResponse(documents=[], success=False, error_message="响应格式异常")
        return RerankResponse(documents=rerank_docs)

    def _rerank_siliconflow(self, query: str, documents: List[str], top_n: Optional[int] = None) -> RerankResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model_name, "query": query, "documents": documents, "return_documents": True}
        if top_n is not None:
            data["top_n"] = top_n
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        rerank_docs = []
        if "results" in result:
            for item in result["results"]:
                document = item.get("document", {})
                content = document.get("text", "") if isinstance(document, dict) else str(document)
                rerank_docs.append(RerankDocument(
                    index=item.get("index", 0), content=content,
                    relevance_score=float(item.get("relevance_score", 0.0))
                ))
        else:
            return RerankResponse(documents=[], success=False, error_message="响应格式异常")
        return RerankResponse(documents=rerank_docs)

    def health_check(self) -> Dict[str, Any]:
        status = {"service": "RerankService", "provider": self.provider.value, "status": "unknown"}
        try:
            r = self.rerank_documents(query="测试", documents=["测试"], top_n=1)
            status["status"] = "healthy" if r.success else "error"
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
        return status


def create_rerank_service() -> Optional[RerankService]:
    try:
        if not Config.RERANK_PROVIDER or not Config.RERANK_API_KEY or not Config.RERANK_MODEL_NAME:
            return None
        try:
            provider = RerankProvider(Config.RERANK_PROVIDER.lower())
        except ValueError:
            logger.error(f"不支持的重排序提供商: {Config.RERANK_PROVIDER}")
            return None
        return RerankService(
            provider=provider, api_key=Config.RERANK_API_KEY,
            model_name=Config.RERANK_MODEL_NAME, endpoint=Config.RERANK_ENDPOINT
        )
    except Exception:
        return None


_rerank_service_instance = None


def get_rerank_service() -> Optional[RerankService]:
    global _rerank_service_instance
    if _rerank_service_instance is None:
        _rerank_service_instance = create_rerank_service()
    return _rerank_service_instance

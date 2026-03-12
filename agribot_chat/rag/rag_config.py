"""
RAG 服务专用配置

加载 .env 顺序：agribot_chat/.env -> rag/.env
可与 agribot_admin 共用同一 .env（复制或软链接到 agribot_chat/ 或 rag/）
"""
import os
from dotenv import load_dotenv

_dir = os.path.dirname(os.path.abspath(__file__))
# 优先 agribot_chat 目录，否则尝试项目根目录
load_dotenv(os.path.join(_dir, ".env"))
load_dotenv(os.path.join(os.path.dirname(_dir), ".env"))


class Config:
    """RAG 服务配置类"""

    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER = os.getenv("MILVUS_USER") or ""
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD") or ""
    MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "agribot")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "agribot_collection")
    MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
    MILVUS_VECTOR_DIM = int(os.getenv("MILVUS_VECTOR_DIM", "1024"))

    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

    LLM_EMBEDDING_MODEL_NAME = os.getenv("LLM_EMBEDDING_MODEL_NAME", "")
    LLM_EMBEDDING_API_KEY = os.getenv("LLM_EMBEDDING_API_KEY", "")
    LLM_EMBEDDING_BASE_URL = os.getenv("LLM_EMBEDDING_BASE_URL", "")

    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER")
    RERANK_API_KEY = os.getenv("RERANK_API_KEY")
    RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME")
    RERANK_ENDPOINT = os.getenv("RERANK_ENDPOINT")

    # Neo4j（Graph RAG）
    NEO4J_URI = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

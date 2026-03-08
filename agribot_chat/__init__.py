"""
agribot_chat：用户聊天 / RAG 知识库问答独立模块

与 agribot_admin、agribot_index 同级，提供 RAG 问答服务。
"""
from agribot_chat.rag.rag_service import get_rag_service, RAGResponse

__all__ = ["get_rag_service", "RAGResponse"]

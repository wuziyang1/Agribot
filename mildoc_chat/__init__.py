"""
mildoc_chat：用户聊天 / RAG 知识库问答独立模块

与 mildoc_admin、mildoc_index 同级，提供 RAG 问答服务。
"""
from mildoc_chat.rag_service import get_rag_service, RAGResponse

__all__ = ["get_rag_service", "RAGResponse"]

"""
RAG服务工具类 (基于LangChain实现)

使用LangChain + Milvus实现RAG服务
从Milvus向量数据库检索相关文档并通过大模型生成回答

作者：开发工程师
日期：2025年01月
"""

'''这个才是llm rag，这个才是真正的rag核心'''
import logging
import queue
import threading
from typing import Iterable, List, Optional, Dict, Any

from pydantic import BaseModel
from pymilvus import MilvusClient  # 直接使用 pymilvus，避免 langchain-milvus 黑盒导致 0 结果
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from mildoc_chat.rag.rag_config import Config
from mildoc_chat.rag.rerank_service import get_rerank_service

# 配置日志
logger = logging.getLogger(__name__)


class SourceDocument(BaseModel):
    """源文档信息模型"""
    doc_name: str  # 文档名称
    doc_path_name: str  # 文档完整路径
    doc_type: str  # 文档类型
    content_preview: str  # 内容预览（前200字符）
    similarity_score: Optional[float] = None  # 相似度分数


class TokenUsage(BaseModel):
    """Token使用情况模型"""
    prompt_tokens: int  # 输入token数
    completion_tokens: int  # 输出token数
    total_tokens: int  # 总token数


class RAGResponse(BaseModel):
    """RAG服务响应模型"""
    content: str  # 大模型回复给用户的文本内容
    source_documents: List[SourceDocument]  # 检索使用的源文档列表
    token_usage: Optional[TokenUsage] = None  # token使用情况
    success: bool = True  # 查询是否成功
    error_message: Optional[str] = None  # 错误信息
    scene_info: Optional[Dict[str, Any]] = None  # 场景检测信息


class RAGService:
    """RAG服务类 (基于LangChain实现)
    
    使用LangChain + Milvus向量数据库实现检索增强生成服务
    支持OpenAI兼容的大模型和嵌入模型
    """
    
    # 场景检测提示词模板
    SCENE_DETECTION_TEMPLATE = """
    请分析用户问题属于以下哪种客服场景类型，只返回对应的数字：

        1. 产品咨询类 - 询问产品功能、规格、价格等基本信息
        2. 售后服务类 - 退换货、维修、质量问题等售后相关
        3. 账户相关类 - 登录、注册、密码、个人信息等账户问题  
        4. 投诉建议类 - 对服务或产品的投诉、意见、建议
        5. 技术支持类 - 使用方法、故障排除、技术配置等
        6. 其他咨询类 - 不属于以上分类的一般性咨询

    用户问题: {question}    

    请只返回场景类型对应的数字（1-6）：
    """
    
    # 统一的提示词模板 - 通用知识库问答版本
    PROMPT_TEMPLATE = """
    你是一个基于企业文档知识库的问答助手，优先利用检索到的文档内容来回答用户问题，但在知识库没有覆盖时可以结合通用常识进行补充说明。

    知识库内容（可能来自多个文档片段，可能为空或不相关）:
    {context}

    用户问题: {question}

    回答要求：
        1. 当上方“知识库内容”中存在与问题明显相关的信息时，应以这些内容为主进行回答，对关键信息做整理、概括和重组。
        2. 当知识库中的信息只覆盖了问题的一部分时，先基于已有内容回答“已知部分”，然后可以适度结合通用经验补充，但不要与知识库中已有结论相矛盾。
        3. 当知识库内容与问题几乎无关或基本为空时，可以直接基于通用知识/常识完整回答用户问题，此时不必刻意强调“知识库里查不到”，但也不要虚构具体文档中才会出现的细节（如具体公司名称、条款编号等）。
        4. 回答使用自然流畅的中文，语言简洁、逻辑清晰，可以合理使用 Markdown 语法（如标题、列表、加粗、代码块等）提升可读性。

    请基于以上要求，使用 Markdown 格式给出对用户问题的最终回答：
    """
    
    def __init__(self):
        """初始化RAG服务"""
        self.milvus_client: Optional[MilvusClient] = None
        self.embeddings = None
        self.llm = None
        self.rerank_service = None  # 重排序服务
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 初始化嵌入模型
            self._initialize_embeddings()
            
            # 初始化大语言模型
            self._initialize_llm()
            
            # 初始化向量存储（基于 pymilvus 客户端）
            self._initialize_vector_store()
            
            # 初始化重排序服务
            self._initialize_rerank_service()
            
            logger.info("RAG服务初始化完成")
            
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {e}")
            raise
    
    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 使用自定义嵌入类，兼容OpenAI API
            from openai import OpenAI
            
            class CustomEmbeddings:
                def __init__(self, model_name: str, api_key: str, api_base: str, dimensions: int):
                    self.model_name = model_name
                    self.client = OpenAI(api_key=api_key, base_url=api_base)
                    self.dimensions = dimensions
                
                def embed_query(self, text: str) -> List[float]:
                    """嵌入单个查询"""
                    return self.embed_documents([text])[0]
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    """嵌入多个文档"""
                    try:
                        response = self.client.embeddings.create(
                            model=self.model_name,
                            input=texts,
                            dimensions=self.dimensions,
                            encoding_format="float"
                        )
                        return [data.embedding for data in response.data]
                    except Exception as e:
                        logger.error(f"嵌入生成失败: {e}")
                        raise
            
            self.embeddings = CustomEmbeddings(
                model_name=Config.LLM_EMBEDDING_MODEL_NAME,
                api_key=Config.LLM_EMBEDDING_API_KEY,
                api_base=Config.LLM_EMBEDDING_BASE_URL,
                dimensions=Config.MILVUS_VECTOR_DIM
            )
            
            # 测试嵌入模型
            test_embedding = self.embeddings.embed_query("测试")
            actual_dim = len(test_embedding)
            
            logger.info(f"嵌入模型初始化成功: {Config.LLM_EMBEDDING_MODEL_NAME}")
            logger.info(f"向量维度: {actual_dim}")
            
            if actual_dim != Config.MILVUS_VECTOR_DIM:
                logger.warning(f"向量维度不匹配! 实际({actual_dim}) != 期望({Config.MILVUS_VECTOR_DIM})")
            
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise
    
    def _initialize_llm(self):
        """初始化大语言模型"""
        try:
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL_NAME,
                openai_api_key=Config.LLM_API_KEY,
                openai_api_base=Config.LLM_BASE_URL,
                temperature=0.1,
                max_tokens=800  # 调整为800，平衡详细度和简洁性
            )
            
            logger.info(f"大语言模型初始化成功: {Config.LLM_MODEL_NAME}")
            
        except Exception as e:
            logger.error(f"大语言模型初始化失败: {e}")
            raise

    def _create_llm(self, *, streaming: bool, callbacks: Optional[list] = None) -> ChatOpenAI:
        """创建 LLM 客户端（按需开启 streaming）"""
        kwargs: Dict[str, Any] = {
            "model": Config.LLM_MODEL_NAME,
            "openai_api_key": Config.LLM_API_KEY,
            "openai_api_base": Config.LLM_BASE_URL,
            "temperature": 0.1,
            "max_tokens": 800,
        }
        if streaming:
            kwargs["streaming"] = True
        if callbacks:
            kwargs["callbacks"] = callbacks
        return ChatOpenAI(**kwargs)
    
    def _initialize_vector_store(self):
        """初始化向量存储（pymilvus 客户端）

        说明：
          - 这里直接使用 pymilvus.MilvusClient，而不再依赖 langchain-milvus 的封装，
            避免因为连接别名 / 默认配置等问题导致“集合里明明有数据但始终检索到 0 条”的情况。
          - mildoc_index 构建集合时的字段是：
              id, doc_name, doc_path_name, doc_type, doc_md5, doc_length, content, content_vector, embedding_model
        """
        try:
            # 显式使用 uri + db_name，直连 mildoc_index 使用的 Milvus 实例
            uri = f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}"
            self.milvus_client = MilvusClient(
                uri=uri,
                db_name=Config.MILVUS_DATABASE or "default",
                user=Config.MILVUS_USER or "",
                password=Config.MILVUS_PASSWORD or "",
            )

            # 做一次轻量级的探测，确认集合存在且可搜索
            try:
                stats = self.milvus_client.get_collection_stats(
                    collection_name=Config.MILVUS_COLLECTION_NAME
                )
                row_count = int(stats.get("row_count", -1))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"获取集合统计信息失败: {e}")
                row_count = -1

            logger.info(
                "Milvus向量存储初始化成功: %s (uri=%s db=%s rows=%s)",
                Config.MILVUS_COLLECTION_NAME,
                uri,
                Config.MILVUS_DATABASE,
                row_count,
            )
            
        except Exception as e:
            logger.error(f"Milvus向量存储初始化失败: {e}")
            raise
    


    def _initialize_rerank_service(self):
        """初始化重排序服务"""
        try:
            self.rerank_service = get_rerank_service()
            if self.rerank_service:
                logger.info("重排序服务初始化成功")
            else:
                logger.info("重排序服务未配置，将跳过rerank步骤")
        except Exception as e:
            logger.warning(f"重排序服务初始化失败: {e}")
            self.rerank_service = None

    def query_service(self, query: str, use_rerank: bool = True, use_rag: bool = True) -> RAGResponse:
        """核心查询服务方法
        
        Args:
            query: 用户输入的查询内容
            use_rerank: 是否使用重排序功能
            use_rag: 是否使用知识库检索（为 False 时不查库，仅用 LLM 回答）
            
        Returns:
            RAGResponse: 包含回答内容、源文档和token使用情况的响应对象
        """
        try:
            logger.info(f"🔍 开始处理查询（use_rag={use_rag}, rerank={use_rerank}): {query}")
            
            if not query or not query.strip():
                return RAGResponse(
                    content="请输入有效的查询内容",
                    source_documents=[],
                    success=False,
                    error_message="查询内容为空"
                )
            
            # 第0步：场景检测（可选）
            # scene_info = self.detect_user_scene(query)
            scene_info = None # 暂时不使用场景检测
            
            # 第一步：向量检索获取候选文档（use_rag=False 时跳过，不查库）
            initial_k = 10 if use_rerank and self.rerank_service else 3  # 如果启用重排就检索10个，不启用就检索3个
            candidate_docs: List[Any] = []
            if use_rag and self.milvus_client is not None:
                try:
                    # 1. 先对查询做向量化
                    query_vector = self.embeddings.embed_query(query)

                    # 2. 调用 Milvus 相似度搜索
                    search_params = {
                        "metric_type": "COSINE",
                        "params": {"nprobe": 64},
                    }
                    results = self.milvus_client.search(
                        collection_name=Config.MILVUS_COLLECTION_NAME,
                        data=[query_vector],
                        anns_field="content_vector",
                        search_params=search_params,
                        limit=initial_k,
                        output_fields=[
                            "doc_name",
                            "doc_path_name",
                            "doc_type",
                            "content",
                        ],
                    )

                    hits = results[0] if results else []
                    for hit in hits:
                        # hit 结构: {'id': ..., 'distance': ..., 'entity': {...}}
                        entity = hit.get("entity", {})
                        page_content = entity.get("content", "")
                        metadata = {
                            "doc_name": entity.get("doc_name", ""),
                            "doc_path_name": entity.get("doc_path_name", ""),
                            "doc_type": entity.get("doc_type", ""),
                            "score": float(hit.get("distance", 0.0)),
                        }

                        # 简单的对象，后面当成有 .page_content / .metadata 属性的对象使用
                        class _Doc:
                            def __init__(self, content, meta):
                                self.page_content = content
                                self.metadata = meta

                        candidate_docs.append(_Doc(page_content, metadata))

                except Exception as se:  # noqa: BLE001
                    logger.error(f"向量检索失败: {se}")
            elif use_rag and self.milvus_client is None:
                logger.error("Milvus 客户端尚未初始化")

            logger.info(f"📄 初始检索到 {len(candidate_docs)} 个候选文档")

            # 第二步：重排序（如果启用）
            final_docs = candidate_docs
            if use_rerank and self.rerank_service and len(candidate_docs) > 1: #候选文档数量大于1
                # 提取文档内容用于重排序
                doc_contents = [doc.page_content for doc in candidate_docs]
                
                # 增加重排序的top_n数量，确保不会过滤掉高相关度文档
                rerank_top_n = min(5, len(candidate_docs))  #注意rerank_top_n只是个数字
                #如果候选文档是10个，那么就从这10个里面里面挑出来5个。 重排序服务计算成本较高，不需要对所有10个文档都重排序，选择前5个即可
                #如果是3个，就直接返回3个
                
                # 执行重排序
                #rerank_service是rerank_service.py中的RerankService类的对象，rerank_documents是RerankService类中的方法
                rerank_response = self.rerank_service.rerank_documents(
                    query=query,
                    documents=doc_contents,
                    top_n=rerank_top_n
                )
                '''
                rerank_response的结构是：
                class RerankDocument(BaseModel):
                    """重排序文档模型"""
                    index: int  # 原始文档索引。 原始文档在输入列表中的位置
                    content: str  # 文档内容
                    relevance_score: float  # 相关性分数 重排序后的相关性分数
                    metadata: Optional[Dict[str, Any]] = None  # 元数据
                '''
                
                if rerank_response.success:
                    # 重排序服务返回的结果与原始文档对象进行映射和整合。
                    #重排服务返回的数据类型是RerankDocument，但我们需要的是 原始的 Document 对象， 因为它包含完整的元数据。
                    reranked_docs = []
                    for rerank_doc in rerank_response.documents:
                        if 0 <= rerank_doc.index < len(candidate_docs):
                            original_doc = candidate_docs[rerank_doc.index]
                            # 将相关性分数添加到元数据中
                            if hasattr(original_doc, 'metadata'):
                                original_doc.metadata['rerank_score'] = rerank_doc.relevance_score
                            reranked_docs.append(original_doc)
                    
                    # 安全检查：确保原始最高相似度文档不会被完全过滤掉
                        # 向量检索的第1个文档（最高相似度）在重排序后可能被排到后面，甚至被过滤掉
                        # 如果重排序服务只返回 top 5，而原始第1个文档排到了第6位或更后，它就不会出现在结果中，下面的代码就是 将其添加到结果中
                    if candidate_docs and len(reranked_docs) > 0:
                        first_doc = candidate_docs[0] #去处重拍前检索到的第一个文档
                        first_doc_in_rerank = any( #通过文档名称和内容判断原始第1个文档是否已在重排序结果中。
                            hasattr(doc, 'metadata') and  #hasattr 检查对象是否有指定的属性或方法
                            doc.metadata.get('doc_name') == first_doc.metadata.get('doc_name') and
                            doc.page_content == first_doc.page_content
                            for doc in reranked_docs
                        )
                        
                        if not first_doc_in_rerank: #如果不在，强制添加
                            # 将原始最高相似度文档添加到重排序结果的开头
                            if hasattr(first_doc, 'metadata'):
                                first_doc.metadata['rerank_score'] = 1.0  # 给予最高分数
                            reranked_docs.insert(0, first_doc)
                            logger.info(f"🔒 安全检查：将原始最高相似度文档添加到重排序结果中")
                    
                    final_docs = reranked_docs[:3]  # 最终仍然只取前3个
                    logger.info(f"🔄 重排序完成，选择了 {len(final_docs)} 个文档")
                else:
                    logger.warning(f"重排序失败，使用原始检索结果: {rerank_response.error_message}")
            
            # 第三步：使用选定的文档生成回答
            with get_openai_callback() as cb:  ## 在上下文中获取 OpenAI 回调处理器，方便地公开令牌和成本信息
                # 构建上下文
                context = "\n\n".join([doc.page_content for doc in final_docs])
                
                # 使用统一的提示模板生成回答
                prompt = self.PROMPT_TEMPLATE.format(context=context, question=query)
                answer = self.llm.invoke(prompt).content
            
            # 更新文档引用为最终选择的文档
            source_documents = final_docs #final_docs是第二步最终返回的结果
            
            logger.info(f"✅ 查询完成，检索到 {len(source_documents)} 个相关文档")#这个可以在第二步就打印出来
            logger.info(f"📄 答案长度: {len(answer)} 字符")
            logger.info(f"💰 Token使用: 输入{cb.prompt_tokens}, 输出{cb.completion_tokens}, 总计{cb.total_tokens}")
            
            # 处理源文档信息
            processed_source_docs = []
            for i, doc in enumerate(source_documents):
                try:
                    # 提取文档元数据
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    doc_name = metadata.get("doc_name", f"文档{i+1}")
                    doc_path_name = metadata.get("doc_path_name", "")
                    doc_type = metadata.get("doc_type", "unknown")
                    rerank_score = metadata.get("rerank_score")
                    
                    # 获取内容预览
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    
                    source_doc = SourceDocument(
                        doc_name=doc_name,
                        doc_path_name=doc_path_name,
                        doc_type=doc_type,
                        content_preview=content_preview,
                        similarity_score=rerank_score  # 使用rerank分数
                    )
                    processed_source_docs.append(source_doc)
                    
                    score_info = f" (rerank: {rerank_score:.3f})" if rerank_score else ""
                    logger.info(f"📖 文档{i+1}: {doc_name}{score_info} - {content_preview[:50]}...")
                    
                except Exception as e:
                    logger.warning(f"处理源文档{i+1}时出错: {e}")
                    # 添加默认文档信息
                    processed_source_docs.append(SourceDocument(
                        doc_name=f"文档{i+1}",
                        doc_path_name="",
                        doc_type="unknown",
                        content_preview="无法获取文档信息"
                    ))
            
            # 构建token使用情况
            token_usage = TokenUsage(
                prompt_tokens=cb.prompt_tokens,
                completion_tokens=cb.completion_tokens,
                total_tokens=cb.total_tokens
            )
            
            return RAGResponse(
                content=answer if answer else "抱歉，我无法根据现有信息回答您的问题。",
                source_documents=processed_source_docs,
                token_usage=token_usage,
                success=True,
                scene_info=scene_info
            )
            
        except Exception as e:
            logger.error(f"❌ 查询服务失败: {e}")
            return RAGResponse(
                content="",
                source_documents=[],
                success=False,
                error_message=f"查询过程中发生错误：{str(e)}",
                scene_info=None
            )

    def stream_query(self, query: str, use_rerank: bool = True, use_rag: bool = True) -> Iterable[Dict[str, Any]]:
        """真正的流式输出（token 级别），逐条 yield 事件字典。

        事件格式：
          - {"type": "start"}
          - {"type": "chunk", "data": {"content": "<token>"}}
          - {"type": "end", "data": <RAGResponse dict>}
          - {"type": "error", "data": <RAGResponse dict>}
        """
        # 兼容不同 LangChain 版本的回调基类路径
        try:  # pragma: no cover
            from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore
        except Exception:  # noqa: BLE001
            from langchain.callbacks.base import BaseCallbackHandler  # type: ignore

        def _resp_to_dict(resp: RAGResponse) -> Dict[str, Any]:
            return {
                "success": resp.success,
                "content": resp.content,
                "error_message": resp.error_message,
                "source_documents": [
                    {
                        "doc_name": d.doc_name,
                        "doc_path_name": d.doc_path_name,
                        "doc_type": d.doc_type,
                        "content_preview": d.content_preview,
                        "similarity_score": d.similarity_score,
                    }
                    for d in (resp.source_documents or [])
                ],
                "token_usage": {
                    "prompt_tokens": resp.token_usage.prompt_tokens,
                    "completion_tokens": resp.token_usage.completion_tokens,
                    "total_tokens": resp.token_usage.total_tokens,
                }
                if resp.token_usage
                else None,
                "scene_info": resp.scene_info,
            }

        yield {"type": "start"}

        if not query or not query.strip():
            resp = RAGResponse(
                content="",
                source_documents=[],
                success=False,
                error_message="查询内容为空",
                scene_info=None,
            )
            yield {"type": "error", "data": _resp_to_dict(resp)}
            return

        # 第0步：场景检测（当前不启用，保持与 query_service 一致）
        scene_info = None

        # 第一步：向量检索获取候选文档（use_rag=False 时跳过，不查库）
        initial_k = 10 if use_rerank and self.rerank_service else 3
        candidate_docs: List[Any] = []

        if use_rag and self.milvus_client is not None:
            try:
                query_vector = self.embeddings.embed_query(query)
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 64},
                }
                results = self.milvus_client.search(
                    collection_name=Config.MILVUS_COLLECTION_NAME,
                    data=[query_vector],
                    anns_field="content_vector",
                    search_params=search_params,
                    limit=initial_k,
                    output_fields=[
                        "doc_name",
                        "doc_path_name",
                        "doc_type",
                        "content",
                    ],
                )

                hits = results[0] if results else []
                for hit in hits:
                    entity = hit.get("entity", {})
                    page_content = entity.get("content", "")
                    metadata = {
                        "doc_name": entity.get("doc_name", ""),
                        "doc_path_name": entity.get("doc_path_name", ""),
                        "doc_type": entity.get("doc_type", ""),
                        "score": float(hit.get("distance", 0.0)),
                    }

                    class _Doc:
                        def __init__(self, content, meta):
                            self.page_content = content
                            self.metadata = meta

                    candidate_docs.append(_Doc(page_content, metadata))

            except Exception as se:  # noqa: BLE001
                logger.error(f"向量检索失败: {se}")
        elif use_rag and self.milvus_client is None:
            logger.error("Milvus 客户端尚未初始化")

        # 第二步：重排序（如果启用）
        final_docs = candidate_docs
        if use_rerank and self.rerank_service and len(candidate_docs) > 1:
            doc_contents = [doc.page_content for doc in candidate_docs]
            rerank_top_n = min(5, len(candidate_docs))
            rerank_response = self.rerank_service.rerank_documents(
                query=query,
                documents=doc_contents,
                top_n=rerank_top_n,
            )
            if rerank_response.success:
                reranked_docs = []
                for rerank_doc in rerank_response.documents:
                    if 0 <= rerank_doc.index < len(candidate_docs):
                        original_doc = candidate_docs[rerank_doc.index]
                        if hasattr(original_doc, "metadata"):
                            original_doc.metadata["rerank_score"] = rerank_doc.relevance_score
                        reranked_docs.append(original_doc)

                if candidate_docs and len(reranked_docs) > 0:
                    first_doc = candidate_docs[0]
                    first_doc_in_rerank = any(
                        hasattr(doc, "metadata")
                        and doc.metadata.get("doc_name") == first_doc.metadata.get("doc_name")
                        and doc.page_content == first_doc.page_content
                        for doc in reranked_docs
                    )
                    if not first_doc_in_rerank:
                        if hasattr(first_doc, "metadata"):
                            first_doc.metadata["rerank_score"] = 1.0
                        reranked_docs.insert(0, first_doc)

                final_docs = reranked_docs[:3]
            else:
                logger.warning(f"重排序失败，使用原始检索结果: {rerank_response.error_message}")

        # 处理源文档信息（可提前算好，end 事件里带上）
        processed_source_docs: List[SourceDocument] = []
        for i, doc in enumerate(final_docs):
            try:
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                doc_name = metadata.get("doc_name", f"文档{i+1}")
                doc_path_name = metadata.get("doc_path_name", "")
                doc_type = metadata.get("doc_type", "unknown")
                rerank_score = metadata.get("rerank_score")
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                processed_source_docs.append(
                    SourceDocument(
                        doc_name=doc_name,
                        doc_path_name=doc_path_name,
                        doc_type=doc_type,
                        content_preview=content_preview,
                        similarity_score=rerank_score,
                    )
                )
            except Exception as e:
                logger.warning(f"处理源文档{i+1}时出错: {e}")
                processed_source_docs.append(
                    SourceDocument(
                        doc_name=f"文档{i+1}",
                        doc_path_name="",
                        doc_type="unknown",
                        content_preview="无法获取文档信息",
                    )
                )

        # 第三步：流式生成回答
        context = "\n\n".join([doc.page_content for doc in final_docs])
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=query)

        _STOP = object()
        token_q: "queue.Queue[object]" = queue.Queue()
        state: Dict[str, Any] = {"answer": "", "token_usage": None, "error": None}

        class _TokenQueueCallbackHandler(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # noqa: ANN401
                if token:
                    token_q.put(token)

        def _worker() -> None:
            try:
                cb_handler = _TokenQueueCallbackHandler()
                llm_stream = self._create_llm(streaming=True, callbacks=[cb_handler])
                with get_openai_callback() as cb:
                    result = llm_stream.invoke(prompt)
                content = getattr(result, "content", "") or ""
                state["answer"] = content
                state["token_usage"] = TokenUsage(
                    prompt_tokens=cb.prompt_tokens,
                    completion_tokens=cb.completion_tokens,
                    total_tokens=cb.total_tokens,
                )
            except Exception as e:  # noqa: BLE001
                state["error"] = e
            finally:
                token_q.put(_STOP)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while True:
            item = token_q.get()
            if item is _STOP:
                break
            yield {"type": "chunk", "data": {"content": str(item)}}

        if state["error"] is not None:
            err = state["error"]
            resp = RAGResponse(
                content="",
                source_documents=[],
                success=False,
                error_message=f"查询过程中发生错误：{err}",
                scene_info=None,
            )
            yield {"type": "error", "data": _resp_to_dict(resp)}
            return

        resp = RAGResponse(
            content=state.get("answer") or "",
            source_documents=processed_source_docs,
            token_usage=state.get("token_usage"),
            success=True,
            error_message=None,
            scene_info=scene_info,
        )
        yield {"type": "end", "data": _resp_to_dict(resp)}
    
    def get_similar_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """获取相似文档（用于调试和分析）
        用于直接查看向量检索的结果，不进行重排序和 LLM 生成。
        
        Args:
            query: 查询内容
            top_k: 返回文档数量
            
        Returns:
            List[Dict]: 相似文档列表
        """
        try:
            logger.info(f"🔍 搜索相似文档: {query} (top_k={top_k})")

            if self.milvus_client is None:
                logger.warning("Milvus 客户端未初始化，无法检索")
                return []

            query_vector = self.embeddings.embed_query(query)
            results = self.milvus_client.search(
                collection_name=Config.MILVUS_COLLECTION_NAME,
                data=[query_vector],
                anns_field="content_vector",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
                limit=top_k,
                output_fields=["doc_name", "doc_path_name", "doc_type", "content"],
            )

            hits = results[0] if results else []
            out: List[Dict[str, Any]] = []
            for hit in hits:
                entity = hit.get("entity", {}) or {}
                out.append(
                    {
                        "content": entity.get("content", ""),
                        "metadata": entity,
                        "similarity_score": float(hit.get("distance", 0.0)),
                        "doc_name": entity.get("doc_name", "未知文档"),
                        "doc_path_name": entity.get("doc_path_name", ""),
                        "doc_type": entity.get("doc_type", "unknown"),
                    }
                )

            logger.info(f"✅ 找到 {len(out)} 个相似文档")
            return out
            
        except Exception as e:
            logger.error(f"❌ 获取相似文档失败: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            Dict: 健康状态信息
        """
        status = {
            "service": "RAGService",
            "status": "healthy",
            "components": {},
            "timestamp": None
        }
        
        try:
            from datetime import datetime
            status["timestamp"] = datetime.now().isoformat()
            
            # 检查嵌入模型
            try:
                test_embedding = self.embeddings.embed_query("健康检查")
                status["components"]["embeddings"] = {
                    "status": "healthy",
                    "model": Config.LLM_EMBEDDING_MODEL_NAME,
                    "dimension": len(test_embedding)
                }
            except Exception as e:
                status["components"]["embeddings"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["status"] = "degraded"
            
            # 检查大语言模型
            try:
                with get_openai_callback() as cb:
                    test_response = self.llm.invoke("你好").content
                status["components"]["llm"] = {
                    "status": "healthy",
                    "model": Config.LLM_MODEL_NAME,
                    "response_length": len(test_response) if test_response else 0,
                    "test_tokens": cb.total_tokens
                }
            except Exception as e:
                status["components"]["llm"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["status"] = "degraded"
            
            # 检查向量存储
            try:
                if self.milvus_client is None:
                    raise RuntimeError("Milvus client not initialized")

                query_vector = self.embeddings.embed_query("测试")
                results = self.milvus_client.search(
                    collection_name=Config.MILVUS_COLLECTION_NAME,
                    data=[query_vector],
                    anns_field="content_vector",
                    search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
                    limit=1,
                    output_fields=["doc_name"],
                )
                hits = results[0] if results else []
                status["components"]["vector_store"] = {
                    "status": "healthy",
                    "collection": Config.MILVUS_COLLECTION_NAME,
                    "test_results": len(hits),
                }
            except Exception as e:
                status["components"]["vector_store"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["status"] = "degraded"
            
            # 检查重排序服务
            try:
                if self.rerank_service:
                    rerank_health = self.rerank_service.health_check()
                    status["components"]["rerank_service"] = rerank_health
                else:
                    status["components"]["rerank_service"] = {
                        "status": "not_configured",
                        "message": "重排序服务未配置"
                    }
            except Exception as e:
                status["components"]["rerank_service"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["status"] = "degraded"
            
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
        
        return status

    def detect_user_scene(self, query: str) -> Dict[str, Any]:
        """检测用户问题的场景类型
        
        Args:
            query: 用户问题
            
        Returns:
            Dict: 包含场景类型和建议的字典
        """
        try:
            # 使用LLM进行场景检测
            prompt = self.SCENE_DETECTION_TEMPLATE.format(question=query)
            response = self.llm.invoke(prompt).content.strip()
            
            # 解析场景类型
            scene_mapping = {
                "1": {"type": "产品咨询类", "priority": "normal", "suggest_human": False},
                "2": {"type": "售后服务类", "priority": "high", "suggest_human": True},
                "3": {"type": "账户相关类", "priority": "high", "suggest_human": True},
                "4": {"type": "投诉建议类", "priority": "urgent", "suggest_human": True},
                "5": {"type": "技术支持类", "priority": "normal", "suggest_human": False},
                "6": {"type": "其他咨询类", "priority": "normal", "suggest_human": False}
            }
            
            scene_info = scene_mapping.get(response, scene_mapping["6"])
            scene_info["detected_number"] = response
            
            logger.info(f"🎯 场景检测结果: {query} -> {scene_info['type']} (优先级: {scene_info['priority']})")
            
            return scene_info
            
        except Exception as e:
            logger.warning(f"场景检测失败: {e}")
            return {"type": "其他咨询类", "priority": "normal", "suggest_human": False, "detected_number": "6"}


# 全局RAG服务实例
_rag_service_instance = None

def get_rag_service() -> Optional[RAGService]:
    """获取RAG服务实例（单例模式）"""
    global _rag_service_instance
    
    if _rag_service_instance is None:
        try:
            _rag_service_instance = RAGService()
            logger.info("RAG服务实例创建成功")
        except Exception as e:
            logger.error(f"RAG服务实例创建失败: {e}")
            return None
    
    return _rag_service_instance


# 便捷函数
def query_question(question: str) -> RAGResponse:
    """查询问题的便捷函数
    
    Args:
        question: 用户问题
        
    Returns:
        RAGResponse: 查询响应
    """
    rag_service = get_rag_service()
    if rag_service is None:
        return RAGResponse(
            content="",
            source_documents=[],
            success=False,
            error_message="RAG服务初始化失败"
        )
    
    return rag_service.query_service(question)


if __name__ == "__main__":
    # 测试代码 - 专业客服RAG系统
    rag = get_rag_service()


    # 检测健康状态
    health = rag.health_check()
    print(f"健康状态: {health}")


    # 测试不同场景的客服问题
    test_cases = [
        "帮我介绍一下盗窃罪",  # 其他咨询类
        "我要退货，怎么办理？",    # 售后服务类
        "忘记密码了，如何重置？",  # 账户相关类
        "你们的服务太差了！",      # 投诉建议类
        "产品无法连接WiFi",       # 技术支持类
    ]

    print(100 * "=")
    
    for i, question in enumerate(test_cases, 1):
        print(100 * "*")
        print(f"\n=== 测试案例 {i}: {question} ===")
        
        # 测试场景检测
        scene_info = rag.detect_user_scene(question)
        print(f"🎯 场景类型: {scene_info['type']}")
        print(f"🚨 优先级: {scene_info['priority']}")
        print(f"👤 建议转人工: {'是' if scene_info['suggest_human'] else '否'}")
        
        # 测试完整查询
        response = rag.query_service(question, use_rerank=True)
        print(f"💬 客服回答: {response.content}")
        if response.scene_info:
            print(f"📊 场景信息: {response.scene_info['type']}")
        if response.token_usage:
            print(f"💰 Token使用: {response.token_usage.total_tokens}")
        print(f"📚 参考文档数: {len(response.source_documents)}")
        
    print(100 * "=")
    
    print("\n=== 测试经典产品咨询 ===")
    response = rag.query_service("介绍一下老人与海这本书", use_rerank=True)
    print(f"回答: {response.content}")
    print(f"场景信息: {response.scene_info}")
    print(f"源文档数量: {len(response.source_documents)}")
    if response.token_usage:
        print(f"Token使用: {response.token_usage.total_tokens}")

    for doc in response.source_documents:
        print("--------------------------------")
        print(f"源文档: {doc.doc_name}")
        print(f"相似度分数: {doc.similarity_score}")
        print(f"内容预览: {doc.content_preview[:50]}...")
        
        
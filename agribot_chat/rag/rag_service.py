"""
RAG服务工具类

使用LangChain + Milvus实现RAG服务
从Milvus向量数据库检索相关文档并通过大模型生成回答
"""

'''这个才是llm rag，这个才是真正的rag核心'''
import logging
import queue
import threading
from typing import Iterable, List, Optional, Dict, Any

from pydantic import BaseModel
from pymilvus import MilvusClient  
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from agribot_chat.rag.rag_config import Config
from agribot_chat.rag.rerank_service import get_rerank_service
from agribot_chat.rag.graph_rag_service import get_graph_rag_service

# 配置日志
logger = logging.getLogger(__name__)

#=======================================================
#数据标准化定义
#=======================================================

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
    # 评估用：检索到的完整上下文（仅当 query_service(..., return_contexts=True) 时填充）
    evaluation_contexts: Optional[List[str]] = None



class _Doc:
    """简单的文档对象，拥有 .page_content / .metadata 属性"""
    def __init__(self, content, meta):
        self.page_content = content
        self.metadata = meta


class RAGService:
    """
        RAG服务类
        使用Milvus向量数据库实现检索增强生成服务，通过大模型生成回答
    """
    
    # 提示词模板（融合向量检索 + 知识图谱双路上下文）
    PROMPT_TEMPLATE = """
        你是一个基于企业文档知识库的问答助手，优先利用检索到的文档内容来回答用户问题，但在知识库没有覆盖时可以结合通用常识进行补充说明。

        知识库内容（来自向量相似度检索，可能来自多个文档片段，可能为空或不相关）:
        {context}

        知识图谱关系（来自图数据库的精准检索，展示实体间的上下级/因果/所属等结构化关系，可能为空）:
        {graph_context}

        以下是本对话的近期历史（请结合历史理解并回答当前问题）:
        {chat_history}

        当前用户问题: {question}

        回答要求：
            1. 综合利用上方"知识库内容"和"知识图谱关系"来回答问题。向量检索提供语义相似的段落，图谱提供精准的实体关联与层级结构，两者互补。
            2. 当知识库和图谱都包含相关信息时，以文档原文为主、图谱关系为辅进行融合回答。
            3. 当知识库中的信息只覆盖了问题的一部分时，先基于已有内容回答"已知部分"，然后可以适度结合通用经验补充，但不要与知识库中已有结论相矛盾。
            4. 当知识库和图谱内容都与问题几乎无关或基本为空时，可以直接基于通用知识/常识完整回答用户问题，此时不必刻意强调"知识库里查不到"，但也不要虚构具体文档中才会出现的细节（如具体公司名称、条款编号等）。
            5. 若有近期对话历史，请结合历史上下文理解当前问题，保持回答连贯、不重复已说明过的内容。
            6. 回答使用自然流畅的中文，语言简洁、逻辑清晰，可以合理使用 Markdown 语法（如标题、列表、加粗、代码块等）提升可读性。

        请基于以上要求，使用 Markdown 格式给出对用户问题的最终回答：
    """

    # 对话历史最多保留条数（避免 prompt 过长）
    CHAT_HISTORY_MAX_MESSAGES = 10

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """将对话历史格式化为 prompt 中的一段文字。"""
        if not chat_history:
            return "（无近期历史）"
        lines = []
        # 只取最近 N 条，避免超出上下文
        for msg in chat_history[-self.CHAT_HISTORY_MAX_MESSAGES :]:
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append("用户：" + content)
            else:
                lines.append("助手：" + content)
        return "\n".join(lines) if lines else "（无近期历史）"
    
    def __init__(self):
        """初始化RAG服务"""
        self.milvus_client: Optional[MilvusClient] = None
        self.embeddings = None
        self.llm = None
        self.rerank_service = None  # 重排序服务
        self.graph_rag = None  # 知识图谱服务（双读融合）
        self._initialize_components()
    #=======================================================
    #初始化组件
    #=======================================================
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

            # 初始化知识图谱服务（可选，用于双读融合检索）
            self._initialize_graph_service()
            
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
        """
            初始化向量存储（pymilvus 客户端）
            说明：
            - agribot_index 构建集合时的字段是：
            id, doc_name, doc_path_name, doc_type, doc_md5, doc_length, content, content_vector, embedding_model
        """
        try:
            # 连接数据库
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

    # =====================================================================
    def _initialize_graph_service(self):
        """初始化知识图谱服务（用于双读融合检索）"""
        try:
            self.graph_rag = get_graph_rag_service()
            if self.graph_rag:
                logger.info("知识图谱服务已集成到 RAGService（双读模式）")
            else:
                logger.info("知识图谱服务未配置，将跳过图谱检索")
        except Exception as e:
            logger.warning(f"知识图谱服务初始化失败（不影响向量检索）: {e}")
            self.graph_rag = None

    # 公共私有方法：向量检索、重排序、文档处理、构建 prompt
    # =====================================================================

    def _retrieve_candidates(self, query, use_rerank, use_rag):
        """第一步：向量检索获取候选文档（use_rag=False 时跳过，不查库）"""
        initial_k = 10 if use_rerank and self.rerank_service else 3  # 如果启用重排就检索10个，不启用就检索3个
        candidate_docs = []

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
                    # hit 结构: {'id': ..., 'distance': ..., 'entity': {...}} entity就是上一步results的数据
                    entity = hit.get("entity", {})
                    page_content = entity.get("content", "")
                    metadata = {
                        "doc_name": entity.get("doc_name", ""),
                        "doc_path_name": entity.get("doc_path_name", ""),
                        "doc_type": entity.get("doc_type", ""),
                        "score": float(hit.get("distance", 0.0)),
                    }
                    candidate_docs.append(_Doc(page_content, metadata))

            except Exception as se:  # noqa: BLE001
                logger.error(f"向量检索失败: {se}")
        elif use_rag and self.milvus_client is None:
            logger.error("Milvus 客户端尚未初始化")

        logger.info(f"📄 初始检索到 {len(candidate_docs)} 个候选文档")
        return candidate_docs

    def _rerank_docs(self, query, candidate_docs, use_rerank):
        """第二步：重排序（如果启用）

        对候选文档进行重排序，并做安全检查确保原始最高相似度文档不会被过滤掉。
        最终返回前3个文档。
        """
        final_docs = candidate_docs
        if use_rerank and self.rerank_service and len(candidate_docs) > 1:  # 候选文档数量大于1
            # 提取文档内容用于重排序
            doc_contents = [doc.page_content for doc in candidate_docs]

            # 增加重排序的top_n数量，确保不会过滤掉高相关度文档
            rerank_top_n = min(5, len(candidate_docs))  # 注意rerank_top_n只是个数字
            # 如果候选文档是10个，那么就从这10个里面里面挑出来5个。 重排序服务计算成本较高，不需要对所有10个文档都重排序，选择前5个即可
            # 如果是3个，就直接返回3个

            # 执行重排序
            # rerank_service是rerank_service.py中的RerankService类的对象，rerank_documents是RerankService类中的方法
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
                # 重排服务返回的数据类型是RerankDocument，但我们需要的是 原始的 Document 对象， 因为它包含完整的元数据。
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
                    first_doc = candidate_docs[0]  # 取出重排前检索到的第一个文档
                    first_doc_in_rerank = any(  # 通过文档名称和内容判断原始第1个文档是否已在重排序结果中。
                        hasattr(doc, 'metadata') and  # hasattr 检查对象是否有指定的属性或方法
                        doc.metadata.get('doc_name') == first_doc.metadata.get('doc_name') and
                        doc.page_content == first_doc.page_content
                        for doc in reranked_docs
                    )

                    if not first_doc_in_rerank:  # 如果不在，强制添加
                        # 将原始最高相似度文档添加到重排序结果的开头
                        if hasattr(first_doc, 'metadata'):
                            first_doc.metadata['rerank_score'] = 1.0  # 给予最高分数
                        reranked_docs.insert(0, first_doc)
                        logger.info(f"🔒 安全检查：将原始最高相似度文档添加到重排序结果中")

                final_docs = reranked_docs[:3]  # 最终仍然只取前3个
                logger.info(f"🔄 重排序完成，选择了 {len(final_docs)} 个文档")
            else:
                logger.warning(f"重排序失败，使用原始检索结果: {rerank_response.error_message}")

        return final_docs

    def _process_source_docs(self, final_docs):
        """将检索/重排后的文档对象转换为 SourceDocument 列表，并打印日志。"""
        processed_source_docs = []
        for i, doc in enumerate(final_docs):
            try:
                # 提取文档元数据
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                doc_name = metadata.get("doc_name", f"文档{i+1}")
                doc_path_name = metadata.get("doc_path_name", "")
                doc_type = metadata.get("doc_type", "unknown")
                rerank_score = metadata.get("rerank_score")

                # 获取内容预览
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content

                processed_source_docs.append(SourceDocument(
                    doc_name=doc_name,
                    doc_path_name=doc_path_name,
                    doc_type=doc_type,
                    content_preview=content_preview,
                    similarity_score=rerank_score  # 使用rerank分数
                ))

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
        return processed_source_docs

    def _retrieve_graph_context(self, query):
        """从知识图谱获取结构化关系上下文（双读的图谱检索部分）

        通过 Cypher 查询或关键词模糊匹配，获取与用户问题相关的
        实体关系信息，作为向量检索的补充上下文。
        """
        if not self.graph_rag:
            return ""
        try:
            graph_rag = self.graph_rag
            graph_rag.graph.refresh_schema()
            schema = graph_rag.graph.schema

            # 尝试 LLM 生成 Cypher 精准查询
            cypher = graph_rag._generate_cypher(query, schema)
            results = []
            if cypher:
                try:
                    results = graph_rag.graph.query(cypher)
                    logger.info("图谱 Cypher 查询返回 %d 条结果", len(results))
                except Exception:
                    pass

            # Cypher 无结果时回退到关键词模糊匹配
            if not results:
                results = graph_rag._fallback_search(query)
                if results:
                    logger.info("图谱关键词回退查询返回 %d 条结果", len(results))

            if not results:
                return ""

            return graph_rag._format_graph_results(results)
        except Exception as e:
            logger.warning(f"图谱检索失败（不影响向量检索）: {e}")
            return ""

    def _build_prompt(self, final_docs, query, chat_history=None, graph_context=""):
        """根据最终文档列表和查询构建 LLM prompt（融合向量上下文 + 图谱上下文）。"""
        context = "\n\n".join([doc.page_content for doc in final_docs])
        chat_history_str = self._format_chat_history(chat_history)
        return self.PROMPT_TEMPLATE.format(
            context=context,
            graph_context=graph_context or "（无图谱信息）",
            chat_history=chat_history_str,
            question=query,
        )

    # =====================================================================
    # 对外查询方法
    # =====================================================================

    def query_service(self, query, use_rerank=True, use_rag=True, use_graph=True, chat_history=None, return_contexts=False):
        """核心查询服务方法

        Args:
            query: 用户输入的查询内容
            use_rerank: 是否使用重排序功能
            use_rag: 是否使用知识库检索（为 False 时不查库，仅用 LLM 回答）
            use_graph: 是否使用知识图谱检索（为 False 时不查图谱）
            chat_history: 本会话的近期对话历史 [{"role":"user"|"assistant","content":"..."}, ...]，可选
            return_contexts: 为 True 时，在返回中附带 evaluation_contexts（检索到的完整文本列表），供 RAG 评估使用
        """
        try:
            logger.info(f"🔍 开始处理查询（use_rag={use_rag}, use_graph={use_graph}, rerank={use_rerank}): {query}")

            if not query or not query.strip():
                return RAGResponse(
                    content="请输入有效的查询内容",
                    source_documents=[],
                    success=False,
                    error_message="查询内容为空"
                )

            # 第一步：向量检索获取候选文档
            candidate_docs = self._retrieve_candidates(query, use_rerank, use_rag)

            # 第二步：重排序
            final_docs = self._rerank_docs(query, candidate_docs, use_rerank)

            # 第三步：从知识图谱获取结构化关系上下文
            graph_context = self._retrieve_graph_context(query) if use_graph else ""
            if graph_context:
                logger.info("🔗 图谱检索到关系上下文（%d 字符）", len(graph_context))

            # 第四步：融合向量上下文 + 图谱上下文，生成回答
            prompt = self._build_prompt(final_docs, query, chat_history, graph_context=graph_context)
            with get_openai_callback() as cb:  # 在上下文中获取 OpenAI 回调处理器，方便地公开令牌和成本信息
                answer = self.llm.invoke(prompt).content

            logger.info(f"✅ 查询完成，检索到 {len(final_docs)} 个相关文档")
            logger.info(f"📄 答案长度: {len(answer)} 字符")
            logger.info(f"💰 Token使用: 输入{cb.prompt_tokens}, 输出{cb.completion_tokens}, 总计{cb.total_tokens}")

            # 处理源文档信息
            processed_source_docs = self._process_source_docs(final_docs)

            # 构建token使用情况
            token_usage = TokenUsage(
                prompt_tokens=cb.prompt_tokens,
                completion_tokens=cb.completion_tokens,
                total_tokens=cb.total_tokens
            )

            evaluation_contexts = [doc.page_content for doc in final_docs] if return_contexts else None
            return RAGResponse(
                content=answer if answer else "抱歉，我无法根据现有信息回答您的问题。",
                source_documents=processed_source_docs,
                token_usage=token_usage,
                success=True,
                evaluation_contexts=evaluation_contexts,
            )

        except Exception as e:
            logger.error(f"❌ 查询服务失败: {e}")
            return RAGResponse(
                content="",
                source_documents=[],
                success=False,
                error_message=f"查询过程中发生错误：{str(e)}",
                evaluation_contexts=None,
            )

    def stream_query(self, query, use_rerank=True, use_rag=True, use_graph=True, chat_history=None):
        """流式输出（token 级别），逐条 yield 事件字典。

        与 query_service 共用检索 / 重排 / 文档处理逻辑，仅 LLM 生成阶段改为流式。
        支持传入 chat_history 使模型结合同一会话的历史对话回答。
        """
        # 兼容不同 LangChain 版本的回调基类路径
        try:  # pragma: no cover
            from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore
        except Exception:  # noqa: BLE001
            from langchain.callbacks.base import BaseCallbackHandler  # type: ignore

        def _resp_to_dict(resp):
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
            }

        yield {"type": "start"}

        if not query or not query.strip():
            resp = RAGResponse(
                content="",
                source_documents=[],
                success=False,
                error_message="查询内容为空",
            )
            yield {"type": "error", "data": _resp_to_dict(resp)}
            return

        # 第一步 & 第二步：向量检索 + 重排序（复用公共方法）
        candidate_docs = self._retrieve_candidates(query, use_rerank, use_rag)
        final_docs = self._rerank_docs(query, candidate_docs, use_rerank)

        # 处理源文档信息（提前算好，end 事件里带上）
        processed_source_docs = self._process_source_docs(final_docs)

        # 第三步：从知识图谱获取结构化关系上下文
        graph_context = self._retrieve_graph_context(query) if use_graph else ""
        if graph_context:
            logger.info("🔗 图谱检索到关系上下文（%d 字符）", len(graph_context))

        # 第四步：融合向量上下文 + 图谱上下文，流式生成回答
        prompt = self._build_prompt(final_docs, query, chat_history, graph_context=graph_context)

        _STOP = object()
        token_q = queue.Queue()
        state = {"answer": "", "token_usage": None, "error": None}

        class _TokenQueueCallbackHandler(BaseCallbackHandler):
            def on_llm_new_token(self, token, **kwargs):
                if token:
                    token_q.put(token)

        def _worker():
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
            )
            yield {"type": "error", "data": _resp_to_dict(resp)}
            return

        resp = RAGResponse(
            content=state.get("answer") or "",
            source_documents=processed_source_docs,
            token_usage=state.get("token_usage"),
            success=True,
            error_message=None,
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
    # 简单测试：健康检查
    rag = get_rag_service()
    health = rag.health_check()
    print(f"健康状态: {health}")
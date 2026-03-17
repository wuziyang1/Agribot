#!/usr/bin/env python3
"""
使用 ragas 对 RAG 系统进行评估（混合检索版）。

- 读取 /export/workspace/rag/experiment/eval/data/rag_test_data.json
- 基于 agribot_chat 的 RAGService，但在脚本内部将检索方式改为「向量检索 + BM25 混合检索」
- 下游提示词与回答生成依然完全复用 chat 模块的 RAG 提示模板和 LLM
- 原有 chat 在线问答接口仍然使用原来的向量检索，不受本脚本影响
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple, Dict
from types import SimpleNamespace

from dotenv import load_dotenv

# 路径常量（固定）
_script_dir = os.path.dirname(os.path.abspath(__file__))
# 脚本位于 /export/workspace/rag/experiment/eval
# 项目根目录是再往上一层：/export/workspace/rag
_project_root = os.path.dirname(os.path.dirname(_script_dir))
DATA_PATH = "/export/workspace/rag/experiment/generate_data/gen_data.json"
OUT_PATH = "/export/workspace/rag/experiment/2-bm25_emb/bm25_res.json"

# 加载本目录下的 .env（若有），并保证能导入 agribot_chat
_env_experiment = os.path.join(_script_dir, ".env")
load_dotenv(_env_experiment)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import agribot_chat.rag.rag_config as rag_config
from agribot_chat.rag.rag_service import get_rag_service


def _load_test_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


_GLOBAL_BM25_RETRIEVER = None
_GLOBAL_BM25_KEY2META: Dict[str, Dict] = {}


def _build_bm25_key(doc_name: str, content: str) -> str:
    """为文档构造一个稳定的 key，用于在向量检索和 BM25 结果之间对齐。"""
    prefix = (content or "")[:64]
    return f"{doc_name}||{prefix}"


def _ensure_global_bm25(rag, k_bm25: int = 50):
    """
    基于 Milvus 全量文档构建一个全局 BM25Retriever（仅在首次调用时构建，后续复用）,
    这样 BM25 就不再局限于向量检索候选集合，而是一个真正独立的检索通道。
    """
    global _GLOBAL_BM25_RETRIEVER, _GLOBAL_BM25_KEY2META

    if _GLOBAL_BM25_RETRIEVER is not None:
        return _GLOBAL_BM25_RETRIEVER

    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from agribot_chat.rag.rag_config import Config

    milvus = getattr(rag, "milvus_client", None)
    if milvus is None:
        print("[混合检索] Milvus 客户端未初始化，无法构建全量 BM25 索引，将退回仅向量检索。")
        return None

    try:
        stats = milvus.get_collection_stats(collection_name=Config.MILVUS_COLLECTION_NAME)
        row_count = int(stats.get("row_count", 0))
    except Exception as e:  # noqa: BLE001
        print(f"[混合检索] 获取集合统计信息失败，无法构建 BM25：{e}")
        return None

    if row_count <= 0:
        print("[混合检索] 向量集合为空，无法构建 BM25 索引。")
        return None

    try:
        # 简单方式：一次性拉取全部文档（仅用于离线评估脚本，行数通常在可接受范围内）
        results = milvus.query(
            collection_name=Config.MILVUS_COLLECTION_NAME,
            filter="",
            output_fields=["doc_name", "doc_path_name", "doc_type", "content"],
            limit=row_count,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[混合检索] 从 Milvus 读取文档失败，无法构建 BM25：{e}")
        return None

    docs_for_bm25: List[Document] = []
    _GLOBAL_BM25_KEY2META.clear()

    for r in results:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        doc_name = r.get("doc_name", "") or ""
        doc_path_name = r.get("doc_path_name", "") or ""
        doc_type = r.get("doc_type", "") or ""

        key = _build_bm25_key(doc_name, content)
        meta = {
            "key": key,
            "doc_name": doc_name,
            "doc_path_name": doc_path_name,
            "doc_type": doc_type,
        }
        _GLOBAL_BM25_KEY2META[key] = {
            "page_content": content,
            "metadata": meta,
        }
        docs_for_bm25.append(Document(page_content=content, metadata=meta))

    if not docs_for_bm25:
        print("[混合检索] 未从 Milvus 中构建出可用文档，BM25 索引为空。")
        return None

    _GLOBAL_BM25_RETRIEVER = BM25Retriever.from_documents(
        docs_for_bm25, k=min(k_bm25, len(docs_for_bm25))
    )
    print(f"[混合检索] 已基于 {len(docs_for_bm25)} 条文档构建全局 BM25 索引。")
    return _GLOBAL_BM25_RETRIEVER


def _hybrid_retrieve_docs(
    rag,
    query: str,
    k_vec: int = 10,
    k_final: int = 3,
    w_bm25: float = 0.5,
    k_bm25: int = 50,
) -> List:
    """
    真正的混合检索：
    - 通道1：向量检索（Milvus）
    - 通道2：BM25（基于全量文档构建的倒排索引）
    两个通道独立并行检索，最后做 rank 融合并截断到 k_final。
    仅用于本脚本评估，不修改 RAGService 的内部实现。
    """
    # 1. 向量检索（不启用原有重排）
    vec_docs: List = rag._retrieve_candidates(query, use_rerank=False, use_rag=True)  # type: ignore[attr-defined]

    # 如果向量检索完全失败，后面仍然可以只用 BM25 兜底
    vec_n = len(vec_docs)

    # 为向量检索结果构造 key，方便与 BM25 结果对齐
    vec_keys: List[str] = []
    for doc in vec_docs:
        meta = getattr(doc, "metadata", {}) or {}
        doc_name = meta.get("doc_name", "") or ""
        key = _build_bm25_key(doc_name, getattr(doc, "page_content", "") or "")
        vec_keys.append(key)

    # 2. 全量 BM25 检索
    bm25 = _ensure_global_bm25(rag, k_bm25=k_bm25)
    bm25_docs = []
    if bm25 is not None:
        try:
            bm25_docs = bm25.invoke(query)[:k_bm25]
        except Exception as e:  # noqa: BLE001
            print(f"[混合检索] BM25 检索失败，将退回仅向量检索：{e}")
            bm25_docs = []

    # 3. 计算 rank 分数
    # 向量检索 rank 分数：rank 越小分数越高
    vec_rank_score: Dict[str, float] = {}
    if vec_n > 0:
        for i, key in enumerate(vec_keys):
            score = 1.0 - (i / max(1, vec_n - 1)) if vec_n > 1 else 1.0
            # 若同一个 key 出现多次，仅保留最高分（最前面的那个）
            if key not in vec_rank_score or score > vec_rank_score[key]:
                vec_rank_score[key] = score

    # BM25 rank 分数
    bm25_rank_score: Dict[str, float] = {}
    m = len(bm25_docs)
    if m > 0:
        for rank, doc in enumerate(bm25_docs):
            meta = getattr(doc, "metadata", {}) or {}
            key = meta.get("key")
            if not key:
                continue
            score = 1.0 - (rank / max(1, m - 1)) if m > 1 else 1.0
            if key not in bm25_rank_score or score > bm25_rank_score[key]:
                bm25_rank_score[key] = score

    # 如果两个通道都空了，直接返回空
    if not vec_rank_score and not bm25_rank_score:
        return []

    # 4. rank 融合：score = (1 - w_bm25)*vec + w_bm25*bm25
    w_vec = 1.0 - w_bm25
    all_keys = set(vec_rank_score.keys()) | set(bm25_rank_score.keys())
    fused_list: List[Tuple[str, float]] = []
    for key in all_keys:
        score = w_vec * vec_rank_score.get(key, 0.0) + w_bm25 * bm25_rank_score.get(key, 0.0)
        fused_list.append((key, score))

    fused_list.sort(key=lambda x: x[1], reverse=True)
    top_keys = [k for k, _ in fused_list[:k_final]]

    # 5. 根据 key 构造最终文档对象列表
    #    - 若该 key 来自向量检索，直接复用原始 _Doc 对象（含完整元数据）
    #    - 若仅来自 BM25，则从全局元数据缓存中构造一个简单对象（page_content/metadata）
    key2_vec_doc: Dict[str, object] = {}
    for doc, key in zip(vec_docs, vec_keys):
        if key not in key2_vec_doc:
            key2_vec_doc[key] = doc

    final_docs: List[object] = []
    for key in top_keys:
        if key in key2_vec_doc:
            final_docs.append(key2_vec_doc[key])
        else:
            meta_entry = _GLOBAL_BM25_KEY2META.get(key)
            if not meta_entry:
                continue
            final_docs.append(
                SimpleNamespace(
                    page_content=meta_entry["page_content"],
                    metadata=meta_entry["metadata"],
                )
            )

    return final_docs


def _run_rag_and_collect_hybrid(questions_and_ground_truth: list[dict]) -> list[dict]:
    """对每个 question 调用『混合检索版 RAG』，收集 answer 与 contexts。"""
    rag = get_rag_service()
    if rag is None:
        raise RuntimeError("RAG 服务初始化失败，请检查 agribot_chat/.env 中 MILVUS / LLM / EMBEDDING 等配置")

    rows = []
    for i, item in enumerate(questions_and_ground_truth):
        question = (item.get("question") or "").strip()
        reference = (item.get("answer") or "").strip()
        if not question:
            continue

        try:
            # 1. 混合检索得到最终文档列表（替代 query_service 内部的向量检索 + 重排）
            final_docs = _hybrid_retrieve_docs(rag, question, k_vec=10, k_final=3, w_bm25=0.5)
            if not final_docs:
                rows.append(
                    {
                        "user_input": question,
                        "reference": reference,
                        "response": "",
                        "retrieved_contexts": [],
                    }
                )
                continue

            # 2. 不使用图谱，仅用混合检索到的上下文构建 prompt 并调用 LLM
            graph_context = ""
            prompt = rag._build_prompt(final_docs, question, chat_history=None, graph_context=graph_context)  # type: ignore[attr-defined]
            answer = rag.llm.invoke(prompt).content  # type: ignore[attr-defined]

            contexts = [doc.page_content for doc in final_docs]

            rows.append(
                {
                    "user_input": question,
                    "reference": reference,
                    "response": (answer or "").strip(),
                    "retrieved_contexts": contexts,
                }
            )
            print(f"  [混合检索 {i+1}/{len(questions_and_ground_truth)}] 已获取 RAG 回答与上下文")
        except Exception as e:
            print(f"  [混合检索 {i+1}] RAG 调用失败: {e}")
            rows.append(
                {
                    "user_input": question,
                    "reference": reference,
                    "response": "",
                    "retrieved_contexts": [],
                }
            )
            continue

    return rows


def main():
    if not os.path.isfile(DATA_PATH):
        print(f"错误: 未找到测试数据 {DATA_PATH}")
        sys.exit(1)

    test_data = _load_test_data(DATA_PATH)
    print(f"【混合检索】加载 {len(test_data)} 条测试数据，开始调用 RAG 收集回答与上下文…")
    rows = _run_rag_and_collect_hybrid(test_data)

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except ImportError as e:
        print(f"请安装 ragas: pip install ragas。错误: {e}")
        sys.exit(1)

    # 构建 ragas 数据集
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            retrieved_contexts=r["retrieved_contexts"],
            response=r["response"],
            reference=r["reference"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    # ragas 打分用的 LLM 和 Embeddings：
    # 默认复用 chat 模块的 Config.LLM_* / LLM_EMBEDDING_*，
    # 如果 experiment/.env 中提供了 EVAL_*，则优先使用 EVAL_* 作为评估专用模型（不会影响 RAG 问答模型）。
    Config = rag_config.Config
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        print("请安装 langchain-openai: pip install langchain-openai")
        sys.exit(1)

    import os as _os

    def _env_or_default(name: str, default: str) -> str:
        v = _os.getenv(name)
        return default if not v else v

    eval_llm_model = _env_or_default("EVAL_LLM_MODEL_NAME", Config.LLM_MODEL_NAME)
    eval_llm_api_key = _env_or_default("EVAL_LLM_API_KEY", Config.LLM_API_KEY)
    eval_llm_base_url = _env_or_default("EVAL_LLM_BASE_URL", Config.LLM_BASE_URL or "")

    eval_emb_model = Config.LLM_EMBEDDING_MODEL_NAME
    eval_emb_api_key = Config.LLM_EMBEDDING_API_KEY
    eval_emb_base_url = Config.LLM_EMBEDDING_BASE_URL or ""

    chat = ChatOpenAI(
        model=eval_llm_model,
        openai_api_key=eval_llm_api_key,
        openai_api_base=eval_llm_base_url or None,
        temperature=0.0,
        max_tokens=1024,
    )
    bge_embeddings = OpenAIEmbeddings(
        model=eval_emb_model,
        openai_api_key=eval_emb_api_key,
        openai_api_base=eval_emb_base_url or None,
    )

    res = evaluate(
        dataset,
        llm=chat,
        embeddings=bge_embeddings,
        show_progress=True,
    )

    # 转成可序列化的结果并写入
    def _to_serializable(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return {k: _to_serializable(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        return obj

    scores = getattr(res, "scores", None) or getattr(res, "dataset_scores", res)
    out = {"scores": _to_serializable(scores)}
    if hasattr(res, "to_pandas"):
        try:
            df = res.to_pandas()
            out["dataset_scores"] = df.to_dict(orient="records")
        except Exception:
            pass
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"【混合检索】评估完成，结果已写入 {OUT_PATH}")
    print("【混合检索】聚合指标:", out.get("scores", out))


if __name__ == "__main__":
    main()


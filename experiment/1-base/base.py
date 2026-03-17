#!/usr/bin/env python3
"""\
mildocchat
使用 ragas 对 RAG 系统进行评估（参数固定版）。

- 读取 /export/workspace/rag/experiment/eval/data/rag_test_data.json
- 复用 agribot_chat 的 RAG 服务：对每个 question 调用 RAG 得到系统回答与检索到的 contexts
- 用 ragas 计算一组默认的 RAG 指标（faithfulness / relevancy / context 等）
- 评估所用模型（LLM、Embedding）与 chat 模块一致，配置见 agribot_chat/.env

运行（在项目根目录）：
  PYTHONPATH=agribot_chat python experiment/eval/run_rag_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 路径常量（固定）
_script_dir = os.path.dirname(os.path.abspath(__file__))
# 仓库根目录：.../experiment/1-base/base.py -> .../
_project_root = str(Path(__file__).resolve().parents[2])
DATA_PATH = "/export/workspace/rag/experiment/generate_data/gen_data.json"
OUT_PATH = "/export/workspace/rag/experiment/1-base/base_res.json"

from dotenv import load_dotenv

# 先加载当前目录的 .env（如果有），再保证能导入 agribot_chat
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


def _run_rag_and_collect(questions_and_ground_truth: list[dict]) -> list[dict]:
    """对每个 question 调用 RAG，收集 answer 与 contexts。"""
    rag = get_rag_service()
    if rag is None:
        raise RuntimeError("RAG 服务初始化失败，请检查 .env 中 MILVUS / LLM / EMBEDDING 等配置")
    rows = []
    for i, item in enumerate(questions_and_ground_truth):
        question = (item.get("question") or "").strip()
        reference = (item.get("answer") or "").strip()
        if not question:
            continue
        try:
            # 评估阶段只关注向量检索 + RAG，不使用图谱，避免依赖 Neo4j / langchain_neo4j
            resp = rag.query_service(
                question,
                use_rerank=True,
                use_rag=True,
                use_graph=False,
                return_contexts=True,
            )
        except Exception as e:
            print(f"  [{i+1}] RAG 调用失败: {e}")
            rows.append({
                "user_input": question,
                "reference": reference,
                "response": "",
                "retrieved_contexts": [],
            })
            continue
        contexts = resp.evaluation_contexts if getattr(resp, "evaluation_contexts", None) else []
        if not contexts and resp.source_documents:
            contexts = [d.content_preview for d in resp.source_documents]
        rows.append({
            "user_input": question,
            "reference": reference,
            "response": (resp.content or "").strip(),
            "retrieved_contexts": contexts or [],
        })
        print(f"  [{i+1}/{len(questions_and_ground_truth)}] 已获取 RAG 回答与上下文")
    return rows


def main():
    if not os.path.isfile(DATA_PATH):
        print(f"错误: 未找到测试数据 {DATA_PATH}")
        sys.exit(1)

    test_data = _load_test_data(DATA_PATH)
    print(f"加载 {len(test_data)} 条测试数据，开始调用 RAG 收集回答与上下文…")
    rows = _run_rag_and_collect(test_data)

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
    # 评估 LLM：优先用 EVAL_LLM_*，否则回退到 chat 模块的 LLM_*；空字符串也视为未配置
    def _env_or_default(name: str, default: str) -> str:
        v = _os.getenv(name)
        return default if not v else v

    eval_llm_model = _env_or_default("EVAL_LLM_MODEL_NAME", Config.LLM_MODEL_NAME)
    eval_llm_api_key = _env_or_default("EVAL_LLM_API_KEY", Config.LLM_API_KEY)
    eval_llm_base_url = _env_or_default("EVAL_LLM_BASE_URL", Config.LLM_BASE_URL or "")

    # 评估 Embedding：直接复用 chat 模块的 embedding 配置
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

    # 不显式指定 metrics，使用 ragas 0.4.x 默认的一组 RAG 评估指标
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
    print(f"评估完成，结果已写入 {OUT_PATH}")
    print("聚合指标:", out.get("scores", out))


if __name__ == "__main__":
    main()

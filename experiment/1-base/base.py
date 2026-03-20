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
from inspect import signature

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
    # 评估阶段的提示词会把 contexts/response/reference 拼进去，过长容易触发网关校验失败（400）
    # 因此这里默认限制得更保守；如需放宽可用环境变量覆盖
    max_contexts = int(os.getenv("EVAL_MAX_CONTEXTS", "3"))
    max_context_chars = int(os.getenv("EVAL_MAX_CONTEXT_CHARS", "900"))
    max_response_chars = int(os.getenv("EVAL_MAX_RESPONSE_CHARS", "1200"))

    def _clip_text(s: str, max_chars: int) -> str:
        if not s:
            return ""
        s = s.strip()
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + "\n...[truncated]"

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
        contexts = [_clip_text(c, max_context_chars) for c in (contexts or []) if c]
        if max_contexts > 0:
            contexts = contexts[:max_contexts]
        rows.append({
            "user_input": question,
            "reference": reference,
            "response": _clip_text((resp.content or "").strip(), max_response_chars),
            "retrieved_contexts": contexts or [],
        })
        print(f"  [{i+1}/{len(questions_and_ground_truth)}] 已获取 RAG 回答与上下文")
    return rows


def main():
    if not os.path.isfile(DATA_PATH):
        print(f"错误: 未找到测试数据 {DATA_PATH}")
        sys.exit(1)

    test_data = _load_test_data(DATA_PATH)
    total_samples = len(test_data)
    if total_samples == 0:
        print("错误: 测试数据为空")
        sys.exit(1)
    print(f"加载 {total_samples} 条测试数据。")

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except ImportError as e:
        print(f"请安装 ragas: pip install ragas。错误: {e}")
        sys.exit(1)

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
    # 提高评估阶段的超时/重试，避免 ragas 的子 job 过慢导致 TimeoutError
    eval_llm_timeout_s = float(_env_or_default("EVAL_LLM_TIMEOUT_SECONDS", "240"))
    eval_llm_retries = int(_env_or_default("EVAL_LLM_MAX_RETRIES", "10"))

    # 评估 Embedding：直接复用 chat 模块的 embedding 配置
    eval_emb_model = Config.LLM_EMBEDDING_MODEL_NAME
    eval_emb_api_key = Config.LLM_EMBEDDING_API_KEY
    eval_emb_base_url = Config.LLM_EMBEDDING_BASE_URL or ""

    chat_kwargs = {
        "model": eval_llm_model,
        "openai_api_key": eval_llm_api_key,
        "openai_api_base": eval_llm_base_url or None,
        "temperature": 0.0,
        # 提高 max_tokens，减少因输出被截断导致的 OutputParserException（缺少字段）
        "max_tokens": int(os.getenv("EVAL_LLM_MAX_TOKENS", "1024")),
        "request_timeout": eval_llm_timeout_s,
        "max_retries": eval_llm_retries,
    }
    chat = ChatOpenAI(**chat_kwargs)
    bge_embeddings = OpenAIEmbeddings(
        model=eval_emb_model,
        openai_api_key=eval_emb_api_key,
        openai_api_base=eval_emb_base_url or None,
    )

    # -------- 分批评估：每批默认 100 条，可通过环境变量 EVAL_BATCH_SIZE 调整 --------
    from math import ceil

    try:
        batch_size = int(os.getenv("EVAL_BATCH_SIZE", "20"))
        if batch_size <= 0:
            raise ValueError
    except ValueError:
        batch_size = 20

    num_batches = ceil(total_samples / batch_size)
    print(f"将按批次评估: batch_size={batch_size}, 共 {num_batches} 批。")

    all_per_sample_records: list[dict] = []
    weighted_metric_sums: dict[str, float] = {}
    total_for_metrics = 0  # 实际参与指标聚合的样本数（过滤 NaN 后）

    try:
        from ragas.run_config import RunConfig
    except Exception:
        RunConfig = None

    # ragas 的单 job timeout 需要覆盖最慢的一次 LLM 调用
    ragas_timeout_s = float(
        os.getenv("EVAL_RAGAS_TIMEOUT_SECONDS", str(max(480.0, eval_llm_timeout_s * 2)))
    )
    ragas_max_retries = int(os.getenv("EVAL_RAGAS_MAX_RETRIES", str(max(6, eval_llm_retries))))
    ragas_max_workers = int(os.getenv("EVAL_RAGAS_MAX_WORKERS", "1"))

    def _evaluate_with_compat(dataset, llm, embeddings):
        base_kwargs = {
            "dataset": dataset,
            "llm": llm,
            "embeddings": embeddings,
            "show_progress": True,
        }
        if RunConfig is not None:
            base_kwargs["run_config"] = RunConfig(
                timeout=ragas_timeout_s,
                max_retries=ragas_max_retries,
                max_workers=ragas_max_workers,
            )

        # 不同 ragas 版本参数存在差异；优先尝试关闭异常抛出，把失败样本记为 NaN
        sig = signature(evaluate)
        if "raise_exceptions" in sig.parameters:
            base_kwargs["raise_exceptions"] = False
        elif "return_executor" in sig.parameters:
            # 旧版本兼容位，不改变行为，仅避免传入未知参数
            pass

        try:
            return evaluate(**base_kwargs)
        except TypeError:
            # 极端兼容兜底：剥离 run_config/raise_exceptions 后重试
            fallback_kwargs = {
                "dataset": dataset,
                "llm": llm,
                "embeddings": embeddings,
                "show_progress": True,
            }
            return evaluate(**fallback_kwargs)

    for bi in range(num_batches):
        start = bi * batch_size
        end = min(total_samples, (bi + 1) * batch_size)
        batch = test_data[start:end]
        print(f"\n=== 批次 {bi+1}/{num_batches}: 样本 {start}–{end-1}（共 {end-start} 条）===")

        rows = _run_rag_and_collect(batch)

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

        # 不显式指定 metrics，使用 ragas 默认的一组 RAG 评估指标
        res = _evaluate_with_compat(dataset=dataset, llm=chat, embeddings=bge_embeddings)

        # 收集每条样本的指标（用于最终汇总）
        batch_records: list[dict] = []
        if hasattr(res, "to_pandas"):
            try:
                df = res.to_pandas()
                batch_records = df.to_dict(orient="records")
            except Exception:
                batch_records = []

        # 若无法从 pandas 拿到 per-sample，则尝试从 res.dataset_scores / scores 中取
        if not batch_records:
            maybe = getattr(res, "dataset_scores", None)
            if maybe:
                if isinstance(maybe, list):
                    batch_records = maybe
                else:
                    batch_records = [maybe]

        all_per_sample_records.extend(batch_records)

        # 基于本批的 per-sample 指标，对数值型字段做汇总，用于后续整体聚合
        for rec in batch_records:
            metric_keys = [
                k for k, v in rec.items()
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))
            ]
            for k in metric_keys:
                weighted_metric_sums[k] = weighted_metric_sums.get(k, 0.0) + float(rec[k])

        total_for_metrics += len(batch_records)
        print(f"批次 {bi+1} 评估完成，记录数: {len(batch_records)}")

    # -------- 聚合所有批次的指标（简单平均）--------
    final_scores: dict[str, float] = {}
    if total_for_metrics > 0:
        for k, s in weighted_metric_sums.items():
            final_scores[k] = s / total_for_metrics

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

    out = {
        "scores": _to_serializable(final_scores),
        "dataset_scores": _to_serializable(all_per_sample_records),
        "total_samples": total_samples,
        "batch_size": batch_size,
        "num_batches": num_batches,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"评估完成，结果已写入 {OUT_PATH}")
    print("聚合指标:", out.get("scores", out))


if __name__ == "__main__":
    main()

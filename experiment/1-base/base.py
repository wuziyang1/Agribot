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

常见问题解决：
1. OutputParserException: 如果遇到 JSON 解析错误，可以：
   - 设置环境变量 EVAL_LLM_TEMPERATURE=0.1（更稳定的输出）
   - 设置 EVAL_LLM_MAX_TOKENS=2048（避免输出截断）
   - 减小 EVAL_BATCH_SIZE（如设为 10 或 5）
   - 使用支持 JSON 模式的模型（如 GPT-4）
   
2. 超时错误：增加 EVAL_LLM_TIMEOUT_SECONDS 和 EVAL_RAGAS_TIMEOUT_SECONDS

3. 内存不足：减小 EVAL_BATCH_SIZE 和 EVAL_MAX_CONTEXTS
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
        # 对齐混合检索脚本：尽量减少解析抖动
        "temperature": float(os.getenv("EVAL_LLM_TEMPERATURE", "0.0")),
        # 提高 max_tokens，减少因输出被截断导致的 OutputParserException（缺少字段）
        "max_tokens": int(os.getenv("EVAL_LLM_MAX_TOKENS", "1024")),
        "request_timeout": eval_llm_timeout_s,
        "max_retries": eval_llm_retries,
    }

    # 说明：
    # ragas 指标内部会要求模型输出特定结构（并用 pydantic 解析）。
    # 强制开启模型的 json_object 模式，可能会让某些指标的输出结构与 ragas 解析器预期不一致。
    # 因此默认不强制 response_format；如你确实需要可通过环境变量开启。
    if os.getenv("EVAL_FORCE_JSON_RESPONSE_FORMAT", "0") == "1":
        if "gpt" in eval_llm_model.lower() or "o1" in eval_llm_model.lower():
            chat_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    
    chat = ChatOpenAI(**chat_kwargs)
    bge_embeddings = OpenAIEmbeddings(
        model=eval_emb_model,
        openai_api_key=eval_emb_api_key,
        openai_api_base=eval_emb_base_url or None,
    )

    # ---- ragas 指标：显式指定，复用混合检索脚本的稳定打分方式 ----
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )

    # ragas 指标重试/严格度配置，减少 OutputParserException
    metrics_max_retries = int(os.getenv("EVAL_METRICS_MAX_RETRIES", "3"))
    for _m in (context_precision, context_recall, faithfulness):
        if hasattr(_m, "max_retries"):
            _m.max_retries = metrics_max_retries

    # answer_relevancy 默认 strictness=3；调低可降低格式问题概率
    answer_relevancy.strictness = int(os.getenv("EVAL_ANSWER_RELEVANCY_STRICTNESS", "1"))

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

    ragas_run_config = None
    if RunConfig is not None:
        ragas_run_config = RunConfig(
            timeout=ragas_timeout_s,
            max_retries=ragas_max_retries,
            max_workers=ragas_max_workers,
        )

    # 添加自定义的 OutputParser 错误处理
    import logging
    logging.getLogger("ragas.executor").setLevel(logging.WARNING)
    
    # 尝试导入并配置 ragas 的异常处理
    try:
        from ragas.exceptions import RagasOutputParserException
        print("检测到 ragas 输出解析异常支持")
    except ImportError:
        RagasOutputParserException = None

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
        except (TypeError, Exception) as e:
            print(f"  警告: 评估遇到错误 {type(e).__name__}: {e}")
            # 极端兼容兜底：剥离 run_config/raise_exceptions 后重试
            fallback_kwargs = {
                "dataset": dataset,
                "llm": llm,
                "embeddings": embeddings,
                "show_progress": True,
            }
            try:
                return evaluate(**fallback_kwargs)
            except Exception as e2:
                print(f"  错误: 评估失败，跳过此批次: {e2}")
                # 返回空结果避免中断
                return type('EmptyResult', (), {'to_pandas': lambda: None, 'dataset_scores': []})()


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

        # 显式指定 metrics，复用混合检索脚本的稳定评价调用
        try:
            eval_kwargs = {
                "dataset": dataset,
                "metrics": [context_precision, context_recall, faithfulness, answer_relevancy],
                "llm": chat,
                "embeddings": bge_embeddings,
                "batch_size": 1,
            }
            if ragas_run_config is not None:
                eval_kwargs["run_config"] = ragas_run_config
            res = evaluate(**eval_kwargs)
        except Exception as e:
            print(f"  警告: 指标评估失败 ({type(e).__name__}: {e})，尝试更稳定指标子集...")
            try:
                eval_kwargs2 = {
                    "dataset": dataset,
                    "metrics": [answer_relevancy, context_precision, context_recall],
                    "llm": chat,
                    "embeddings": bge_embeddings,
                    "batch_size": 1,
                }
                if ragas_run_config is not None:
                    eval_kwargs2["run_config"] = ragas_run_config
                res = evaluate(**eval_kwargs2)
            except Exception as e2:
                print(f"  错误: 指标评估也失败 ({type(e2).__name__}: {e2})，跳过此批次")
                continue

        # 收集每条样本的指标（用于最终汇总）
        batch_records: list[dict] = []
        if hasattr(res, "to_pandas"):
            try:
                df = res.to_pandas()
                batch_records = df.to_dict(orient="records")
            except Exception as e:
                print(f"  警告: 无法转换为 pandas DataFrame: {e}")
                batch_records = []

        # 若无法从 pandas 拿到 per-sample，则尝试从 res.dataset_scores / scores 中取
        if not batch_records:
            maybe = getattr(res, "dataset_scores", None)
            if maybe:
                if isinstance(maybe, list):
                    batch_records = maybe
                else:
                    batch_records = [maybe]

        # 过滤掉包含 NaN 或 None 的无效记录
        valid_records = []
        for rec in batch_records:
            # 检查是否有有效的数值指标
            has_valid_metric = any(
                isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))
                for v in rec.values()
            )
            if has_valid_metric:
                valid_records.append(rec)
            else:
                print(f"  跳过无效样本记录（所有指标为 NaN）")

        all_per_sample_records.extend(valid_records)

        # 基于本批的 per-sample 指标，对数值型字段做汇总，用于后续整体聚合
        for rec in valid_records:
            metric_keys = [
                k for k, v in rec.items()
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))
            ]
            for k in metric_keys:
                weighted_metric_sums[k] = weighted_metric_sums.get(k, 0.0) + float(rec[k])

        total_for_metrics += len(valid_records)
        print(f"批次 {bi+1} 评估完成，有效记录数: {len(valid_records)}/{len(batch_records)}")

    # -------- 聚合所有批次的指标（简单平均）--------
    final_scores: dict[str, float] = {}
    if total_for_metrics > 0:
        for k, s in weighted_metric_sums.items():
            final_scores[k] = s / total_for_metrics

    # 统计评估成功率
    success_rate = (total_for_metrics / total_samples * 100) if total_samples > 0 else 0
    print(f"\n评估统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  成功评估: {total_for_metrics}")
    print(f"  失败样本: {total_samples - total_for_metrics}")
    print(f"  成功率: {success_rate:.1f}%")

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
        "successful_samples": total_for_metrics,
        "failed_samples": total_samples - total_for_metrics,
        "success_rate": success_rate,
        "batch_size": batch_size,
        "num_batches": num_batches,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n评估完成，结果已写入 {OUT_PATH}")
    print("聚合指标:", final_scores)


if __name__ == "__main__":
    main()

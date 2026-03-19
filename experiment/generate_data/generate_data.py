#!/usr/bin/env python3
"""
mildocindex
RAG 测试数据生成脚本

默认从 Milvus 读取已索引的分片。
也可选从 MinIO 取文档再现场解析（--source parse）或从 MinIO 读已分片 JSON（--source chunks）。

数据流说明：
- MinIO：用户上传的原始文档（PDF 等）
- Milvus：agribot_index 解析后的分片（content、doc_path_name 等），脚本默认从这里读
- 脚本：随机选分片 → 取前后文 → 调 Qwen 生成问题 → 输出 question/answer/source_pdf

依赖：
- .env：MinIO 配置；Milvus 配置（用 milvus 时，可复用 agribot_index/.env 的 MILVUS_*）；QWEN_*
- 运行：PYTHONPATH=agribot_index python experiment/generate_rag_test_data.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any
from dotenv import load_dotenv

# 统一加载环境变量（允许缺失，不报错）：
# - 当前工作目录的 .env（方便本地运行）
# - experiment/eval/.env（本脚本同目录）
# - agribot_index/.env（Milvus 等配置）
_script_dir = os.path.dirname(os.path.abspath(__file__))
# 仓库根目录：.../experiment/eval -> .../
_project_root = os.path.dirname(os.path.dirname(_script_dir))
_agribot_index = os.path.join(_project_root, "agribot_index")

load_dotenv()
load_dotenv(os.path.join(_script_dir, ".env"))
load_dotenv(os.path.join(_agribot_index, ".env"))

# 将 agribot_index 加入 path（Milvus 与解析器均来自该模块）
if _agribot_index not in sys.path:
    sys.path.insert(0, _agribot_index)

# ---------- 从 Milvus 读取已索引分片（不解析文档）----------
def get_milvus_api():  # noqa: E402
    """返回 agribot_index 的 MilvusAPI 实例（需已配置 MILVUS_*）。"""
    from milvus_api import MilvusAPI, MilvusDocumentField  # noqa: E402
    return MilvusAPI(), MilvusDocumentField


def collect_chunks_directly_from_milvus(
    milvus_api: Any,
    milvus_field: Any,
) -> list[tuple[str, str, int, str]]:
    """
    直接从 Milvus 读取已索引分片，不再依赖 MinIO。
    返回 [(doc_path_name, doc_name, chunk_index, chunk_text), ...]。
    """
    # Milvus 对 query 的 (offset+limit) 有窗口上限（常见为 16384），需要分页拉取
    results: list[dict[str, Any]] = []
    page_size = 16_384
    offset = 0
    while True:
        try:
            page = milvus_api.client.query(
                collection_name=milvus_api.collection_name,
                filter="",
                output_fields=[
                    "id",
                    milvus_field.CONTENT.value,
                    milvus_field.DOC_NAME.value,
                    milvus_field.DOC_PATH_NAME.value,
                ],
                limit=page_size,
                offset=offset,
            )
        except Exception:
            return []
        if not page:
            break
        results.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    if not results:
        return []

    # 先按 doc_path_name 分组，再在每个文档内按 id 排序，生成 chunk_index
    by_doc: dict[str, list[tuple[int, str, str]]] = {}
    for r in results:
        doc_path = r.get(milvus_field.DOC_PATH_NAME.value) or ""
        if not doc_path:
            continue
        content = r.get(milvus_field.CONTENT.value) or ""
        if not (content and str(content).strip()):
            continue
        doc_name = r.get(milvus_field.DOC_NAME.value) or os.path.basename(doc_path)
        by_doc.setdefault(doc_path, []).append(
            (int(r.get("id", 0)), doc_name, str(content).strip())
        )

    if not by_doc:
        return []

    flat: list[tuple[str, str, int, str]] = []
    for doc_path, items in by_doc.items():
        # 按 id 排序，保证 chunk 顺序与写入时一致
        items.sort(key=lambda x: x[0])
        for idx, (_id, doc_name, content) in enumerate(items):
            flat.append((doc_path, doc_name, idx, content))

    return flat


def get_chunks_for_document(milvus_api: Any, doc_path_name: str, field: Any) -> list[tuple[int, str, str]]:
    """
    从 Milvus 按 doc_path_name 查询该文档全部分片，按 id 排序，返回 [(id, content, doc_name), ...]。
    分片是 agribot_index 解析后写入的，此处只读不解析。
    """
    try:
        # 过滤 doc_path_name 中的引号，避免 filter 语法错误
        safe_path = doc_path_name.replace('"', '\\"')
        results = milvus_api.client.query(
            collection_name=milvus_api.collection_name,
            filter=f'doc_path_name == "{safe_path}"',
            output_fields=["id", field.CONTENT.value, field.DOC_NAME.value],
            limit=10000,
        )
    except Exception:
        return []
    if not results:
        return []
    # 按 id 保证顺序与解析时一致
    rows = [(r["id"], r.get(field.CONTENT.value) or "", r.get(field.DOC_NAME.value) or "") for r in results]
    rows.sort(key=lambda x: x[0])
    return rows


# ---------- Qwen LLM 调用（OpenAI 兼容接口）----------
def get_llm_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai: pip install openai")
    api_key = os.getenv("QWEN_API_KEY")
    base_url = (os.getenv("QWEN_BASE_URL") or "").strip() or None
    model = os.getenv("QWEN_MODEL", "Qwen3.5-397B-A17B")
    if not api_key:
        raise RuntimeError("请在 .env 中配置 QWEN_API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url), model


def call_llm_generate_question(client: Any, model: str, context_before: str, chunk: str, context_after: str) -> str:
    """根据「前文 + 当前分片 + 后文」让 LLM 生成一个可由当前分片回答的问题。"""
    prompt = f"""
        你是一个出题助手。下面是一份文档的连续片段：先是一段「前文」，然后是「当前分片」，最后是「后文」。
        请根据「当前分片」的内容（可参考前后文理解语境），生成一个用户可能提出的、能够由「当前分片」回答的问题。
        要求：只输出这一个问题，不要解释、不要编号、不要多余标点。问题用中文，简洁明确。

        【前文】
        {context_before or '（无）'}

        【当前分片】
        {chunk}

        【后文】
        {context_after or '（无）'}

        请直接输出一个问题：
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text


def call_llm_extract_minimal_answer(client: Any, model: str, question: str, chunk: str) -> str:
    """
    将「问题 + 原始 chunk」再次输入 LLM，让其仅摘录能回答问题的最小必要片段作为标准答案。
    注意：答案必须直接摘录自 chunk 文本（不允许编造、改写）。
    """
    prompt = f"""
你是一个数据标注助手。你将收到一个“问题”和一个“文本块”。
请仅从以下文本块中提取出回答该问题的最小必要片段作为标准答案。

要求：
1) 答案必须直接摘录自文本块，禁止改写、概括、补充或编造。
2) 严禁包含与问题无关的背景描述或前言后语。
3) 答案长度应尽可能简短，但要足以完整回答问题。
4) 只输出答案本身，不要解释、不要编号、不要加引号。

问题：
{question}

文本块：
{chunk}

请输出标准答案：
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    return ((resp.choices[0].message.content or "").strip())


def _call_with_backoff(fn, *, max_retries: int, base_sleep_s: float, max_sleep_s: float):
    """
    对 LLM 调用做指数退避重试（重点处理 429：TPM/RPM 限流）。
    - 不依赖 openai 内置重试的具体实现与版本差异
    - 其他网络抖动/5xx 也可受益
    """
    last_err: Exception | None = None
    sleep_s = max(0.0, float(base_sleep_s))
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            is_rate_limit = ("429" in msg) or ("rate limit" in msg.lower()) or ("TPM" in msg)
            if attempt >= max_retries:
                raise
            # 限流时更保守一点；非限流也做退避避免打爆
            jitter = (0.25 + random.random() * 0.75)  # 0.25~1.0
            wait = min(max_sleep_s, max(0.2, sleep_s) * (2 ** attempt) * jitter)
            if is_rate_limit:
                wait = min(max_sleep_s, wait + 1.0)
            time.sleep(wait)
    if last_err:
        raise last_err
    raise RuntimeError("LLM call failed unexpectedly")


def sample_chunks_with_context(
    flat_chunks: list[tuple[str, str, int, str]],
    count: int,
    context_size: int,
    rng: random.Random,
) -> list[tuple[str, str, int, list[str], str]]:
    """
    从 flat_chunks 中随机抽取 count 条，每条带「同一文档内」前 context_size、后 context_size 的上下文。
    返回 [(doc_path_name, doc_name, chunk_index, list_of_context_chunks, current_chunk), ...]
    同一文档的 chunks 需按 chunk_index 连续，因此先按 (doc_path_name, chunk_index) 分组再抽。
    """
    by_doc: dict[str, list[tuple[int, str]]] = {}
    for doc_path, doc_name, idx, text in flat_chunks:
        key = doc_path
        if key not in by_doc:
            by_doc[key] = []
        by_doc[key].append((idx, text))
    for key in by_doc:
        by_doc[key].sort(key=lambda x: x[0])

    # 为了让问题覆盖不同 PDF：先按文档分层抽样，再在文档间均匀分配剩余名额
    candidates_by_doc: dict[str, list[tuple[str, str, int, tuple[list[str], list[str]], str]]] = {}
    for doc_path, items in by_doc.items():
        doc_name = os.path.basename(doc_path)
        texts = [x[1] for x in items]
        per_doc: list[tuple[str, str, int, tuple[list[str], list[str]], str]] = []
        for i, (idx, text) in enumerate(items):
            start = max(0, i - context_size)
            end = min(len(texts), i + context_size + 1)
            before = texts[start:i]
            after = texts[i + 1:end]
            per_doc.append((doc_path, doc_name, idx, (before, after), text))
        if per_doc:
            rng.shuffle(per_doc)
            candidates_by_doc[doc_path] = per_doc

    doc_paths = list(candidates_by_doc.keys())
    if not doc_paths or count <= 0:
        return []

    # count 小于文档数：随机挑 count 个文档，每个文档取 1 条
    if count <= len(doc_paths):
        chosen_docs = rng.sample(doc_paths, count)
        return [candidates_by_doc[d].pop() for d in chosen_docs]

    # count 大于等于文档数：先保证每个文档至少 1 条
    chosen_docs = doc_paths[:]
    rng.shuffle(chosen_docs)
    picks: list[tuple[str, str, int, tuple[list[str], list[str]], str]] = []
    for d in chosen_docs:
        if candidates_by_doc[d]:
            picks.append(candidates_by_doc[d].pop())

    # 再把剩余名额在文档间轮询补齐，避免大文档垄断
    remaining = count - len(picks)
    while remaining > 0:
        progressed = False
        for d in chosen_docs:
            if remaining <= 0:
                break
            if candidates_by_doc[d]:
                picks.append(candidates_by_doc[d].pop())
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return picks


def _load_existing_json_array(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _append_and_flush_json_array(path: str, new_rows: list[dict[str, Any]]) -> int:
    """
    读取 path 中已有的 JSON 数组并追加 new_rows，然后原子写回。
    返回写回后的总条数。
    """
    if not new_rows:
        return len(_load_existing_json_array(path))
    existing = _load_existing_json_array(path)
    existing.extend(new_rows)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return len(existing)


# ---------- 主流程 ----------
def main() -> None:
    """
    固定参数版
      - count: 3
      - context-size: 3
      - seed: None
      - output: /export/workspace/rag/experiment/generate_data/gen_data.json
    """
    COUNT = 200
    CONTEXT_SIZE = 3
    SEED = None
    OUT_PATH = "/export/workspace/rag/experiment/generate_data/gen_data.json"

    rng = random.Random(SEED) 

    try:
        milvus_api, milvus_field = get_milvus_api()
    except Exception as e:
        print(f"Milvus 初始化失败: {e}")
        print("  请确保 .env 中配置 MILVUS_*（可复制 agribot_index/.env 中的 Milvus 配置）。")
        sys.exit(1)

    print("直接从 Milvus 读取已索引分片…")
    flat = collect_chunks_directly_from_milvus(milvus_api, milvus_field)

    if not flat:
        print("错误: 未得到任何分片")
        sys.exit(1)

    print(f"共得到 {len(flat)} 个分片，将按文档分层抽取 {COUNT} 条生成问题")

    samples = sample_chunks_with_context(flat, COUNT, CONTEXT_SIZE, rng)

    try:
        llm_client, model = get_llm_client()
    except Exception as e:
        print(f"LLM 初始化失败: {e}")
        sys.exit(1)

    # --- 限流保护（避免触发 TPM/RPM 429）---
    # 每个样本会调用 2 次 LLM（问题 + 答案），建议设置一个最小间隔。
    min_interval_s = float(os.getenv("QWEN_MIN_INTERVAL_SEC", "0.6"))
    max_retries = int(os.getenv("QWEN_MAX_RETRIES", "8"))
    backoff_base_s = float(os.getenv("QWEN_BACKOFF_BASE_SEC", "0.8"))
    backoff_max_s = float(os.getenv("QWEN_BACKOFF_MAX_SEC", "30"))
    last_call_ts = 0.0
    FLUSH_EVERY = 64

    def _throttle():
        nonlocal last_call_ts
        now = time.time()
        to_sleep = (last_call_ts + min_interval_s) - now
        if to_sleep > 0:
            time.sleep(to_sleep)

    # 每 64 条落盘一次：避免中途 429/中断导致前面结果丢失
    pending: list[dict[str, Any]] = []
    already = len(_load_existing_json_array(OUT_PATH))
    if already:
        print(f"检测到已有输出文件，当前已存在 {already} 条，将继续追加…")

    for idx, (doc_path, doc_name, chunk_index, (before, after), chunk) in enumerate(samples):
        context_before = "\n\n".join(before) if before else ""
        context_after = "\n\n".join(after) if after else ""

        # 全局节流：尽量把请求分散到时间轴上，降低 TPM/RPM 峰值
        _throttle()

        try:
            question = _call_with_backoff(
                lambda: call_llm_generate_question(llm_client, model, context_before, chunk, context_after),
                max_retries=max_retries,
                base_sleep_s=backoff_base_s,
                max_sleep_s=backoff_max_s,
            )
        except Exception as e:
            print(f"  [{idx+1}/{len(samples)}] 生成问题失败: {e}")
            question = ""

        last_call_ts = time.time()
        _throttle()

        try:
            answer = (
                _call_with_backoff(
                    lambda: call_llm_extract_minimal_answer(llm_client, model, question, chunk),
                    max_retries=max_retries,
                    base_sleep_s=backoff_base_s,
                    max_sleep_s=backoff_max_s,
                )
                if question
                else ""
            )
        except Exception as e:
            print(f"  [{idx+1}/{len(samples)}] 抽取答案失败: {e}")
            answer = ""

        last_call_ts = time.time()

        pending.append({
            "question": question or "(生成失败)",
            "answer": answer or chunk,
            "source_pdf": doc_path,
        })
        print(f"  [{idx+1}/{len(samples)}] 已生成 -> {doc_name} (chunk {chunk_index})")

        if len(pending) >= FLUSH_EVERY:
            total = _append_and_flush_json_array(OUT_PATH, pending)
            print(f"  已写入 {len(pending)} 条（累计 {total} 条）到 {OUT_PATH}")
            pending.clear()

    if pending:
        total = _append_and_flush_json_array(OUT_PATH, pending)
        print(f"已写入剩余 {len(pending)} 条（累计 {total} 条）到 {OUT_PATH}")
    else:
        total = len(_load_existing_json_array(OUT_PATH))
        print(f"生成完成，文件现有 {total} 条：{OUT_PATH}")


if __name__ == "__main__":
    main()

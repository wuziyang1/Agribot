#!/usr/bin/env python3
"""
mildocindex
RAG 测试数据生成脚本

默认从 Milvus 读取已索引的分片（agribot_index 解析后写入的 content），不解析 PDF。
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
from typing import Any

# 加载 experiment 目录下的 .env
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_script_dir, ".env")
if os.path.isfile(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)

# 将 agribot_index 加入 path（Milvus 与解析器均来自该模块）
_project_root = os.path.dirname(_script_dir)
_agribot_index = os.path.join(_project_root, "agribot_index")
if _agribot_index not in sys.path:
    sys.path.insert(0, _agribot_index)

# ---------- 从 Milvus 读取已索引分片（不解析文档）----------
def _ensure_milvus_env() -> None:
    """使用 Milvus 时加载 agribot_index 的 .env，保证 MILVUS_* 已配置。"""
    from dotenv import load_dotenv
    agribot_env = os.path.join(_agribot_index, ".env")
    if os.path.isfile(agribot_env):
        load_dotenv(agribot_env)


def get_milvus_api():  # noqa: E402
    """返回 agribot_index 的 MilvusAPI 实例（需已配置 MILVUS_*）。"""
    from milvus_api import MilvusAPI, MilvusDocumentField  # noqa: E402
    return MilvusAPI(), MilvusDocumentField


def collect_chunks_directly_from_milvus(
    milvus_api: Any,
    milvus_field: Any,
    max_docs: int,
    rng: random.Random,
) -> list[tuple[str, str, int, str]]:
    """
    直接从 Milvus 读取已索引分片，不再依赖 MinIO。
    返回 [(doc_path_name, doc_name, chunk_index, chunk_text), ...]。
    """
    try:
        results = milvus_api.client.query(
            collection_name=milvus_api.collection_name,
            filter="",
            output_fields=[
                "id",
                milvus_field.CONTENT.value,
                milvus_field.DOC_NAME.value,
                milvus_field.DOC_PATH_NAME.value,
            ],
            limit=100_000,
        )
    except Exception:
        return []

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

    # 随机挑选最多 max_docs 个文档
    doc_paths = list(by_doc.keys())
    chosen_paths = rng.sample(doc_paths, min(max_docs, len(doc_paths)))

    flat: list[tuple[str, str, int, str]] = []
    for doc_path in chosen_paths:
        items = by_doc[doc_path]
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

    # 只保留至少有 1 个 chunk 且当前 chunk 前后能取到足够上下文的位置（或至少当前 chunk 存在）
    candidates: list[tuple[str, str, int, list[str], str]] = []
    for doc_path, items in by_doc.items():
        doc_name = os.path.basename(doc_path)
        indices = [x[0] for x in items]
        texts = [x[1] for x in items]
        for i, (idx, text) in enumerate(items):
            start = max(0, i - context_size)
            end = min(len(texts), i + context_size + 1)
            before = texts[start:i]
            after = texts[i + 1:end]
            candidates.append((doc_path, doc_name, idx, (before, after), text))

    if len(candidates) <= count:
        rng.shuffle(candidates)
        return candidates
    return rng.sample(candidates, count)


# ---------- 主流程 ----------
def main() -> None:
    """
    固定参数版本（仅从 Milvus 读取）：
      - count: 30
      - max-docs: 20
      - context-size: 3
      - seed: None
      - output: /export/workspace/rag/experiment/eval/data/rag_eval_result.json
    如需调整，直接改下面常量即可。
    """
    COUNT = 30
    MAX_DOCS = 20
    CONTEXT_SIZE = 3
    SEED = None
    OUT_PATH = "/export/workspace/rag/experiment/eval/data/rag_eval_result.json"

    rng = random.Random(SEED) 

    _ensure_milvus_env()
    try:
        milvus_api, milvus_field = get_milvus_api()
    except Exception as e:
        print(f"Milvus 初始化失败: {e}")
        print("  请确保 .env 中配置 MILVUS_*（可复制 agribot_index/.env 中的 Milvus 配置）。")
        sys.exit(1)

    print("直接从 Milvus 读取已索引分片（不再访问 MinIO）…")
    flat = collect_chunks_directly_from_milvus(milvus_api, milvus_field, MAX_DOCS, rng)

    if not flat:
        print("错误: 未得到任何分片")
        sys.exit(1)

    print(f"共得到 {len(flat)} 个分片，将随机抽取 {COUNT} 条生成问题")

    samples = sample_chunks_with_context(flat, COUNT, CONTEXT_SIZE, rng)

    try:
        llm_client, model = get_llm_client()
    except Exception as e:
        print(f"LLM 初始化失败: {e}")
        sys.exit(1)

    results = []
    for idx, (doc_path, doc_name, chunk_index, (before, after), chunk) in enumerate(samples):
        context_before = "\n\n".join(before) if before else ""
        context_after = "\n\n".join(after) if after else ""
        try:
            question = call_llm_generate_question(llm_client, model, context_before, chunk, context_after)
        except Exception as e:
            print(f"  [{idx+1}/{len(samples)}] 生成问题失败: {e}")
            question = ""
        results.append({
            "question": question or "(生成失败)",
            "answer": chunk,
            "source_pdf": doc_path,
        })
        print(f"  [{idx+1}/{len(samples)}] 已生成 -> {doc_name} (chunk {chunk_index})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已写入 {len(results)} 条到 {OUT_PATH}")


if __name__ == "__main__":
    main()

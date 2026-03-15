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

import argparse
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

from minio import Minio  # noqa: E402


# ---------- MinIO 连接（与 agribot_index 一致）----------
def get_minio_client() -> Minio:
    endpoint = os.getenv("ENDPOINT")
    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")
    if not all([endpoint, access_key, secret_key]):
        raise RuntimeError("请在 .env 中配置 ENDPOINT, ACCESS_KEY, SECRET_KEY")
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )


def get_bucket() -> str:
    bucket = os.getenv("MINIO_BUCKET", "agribot")
    if not bucket:
        raise RuntimeError("请在 .env 中配置 MINIO_BUCKET")
    return bucket


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


def collect_chunks_from_milvus(
    minio_client: Minio,
    bucket: str,
    milvus_api: Any,
    milvus_field: Any,
    max_docs: int,
    rng: random.Random,
) -> list[tuple[str, str, int, str]]:
    """
    与 agribot_index 一致：从 MinIO 列举文档路径，只保留在 Milvus 中已索引的文档，
    再从 Milvus 读取这些文档的全部分片，展开为 (doc_path_name, doc_name, chunk_index, chunk_text)。
    不做任何 PDF/文档解析。
    """
    object_names = list_document_objects(minio_client, bucket)
    # 只保留 Milvus 里已有索引的文档
    doc_paths_in_milvus = []
    for name in object_names:
        try:
            if milvus_api.check_document_exists(name):
                doc_paths_in_milvus.append(name)
        except Exception:
            continue
    if not doc_paths_in_milvus:
        return []
    chosen = rng.sample(doc_paths_in_milvus, min(max_docs, len(doc_paths_in_milvus)))
    flat = []
    for doc_path in chosen:
        rows = get_chunks_for_document(milvus_api, doc_path, milvus_field)
        doc_name = os.path.basename(doc_path)
        for i, (_id, content, doc_name_from_milvus) in enumerate(rows):
            if not (content and content.strip()):
                continue
            name_use = doc_name_from_milvus or doc_name
            flat.append((doc_path, name_use, i, content.strip()))
    return flat


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
    prompt = f"""你是一个出题助手。下面是一份文档的连续片段：先是一段「前文」，然后是「当前分片」，最后是「后文」。
请根据「当前分片」的内容（可参考前后文理解语境），生成一个用户可能提出的、能够由「当前分片」回答的问题。
要求：只输出这一个问题，不要解释、不要编号、不要多余标点。问题用中文，简洁明确。

【前文】
{context_before or '（无）'}

【当前分片】
{chunk}

【后文】
{context_after or '（无）'}

请直接输出一个问题："""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text


# ---------- 从 MinIO 读取已分片数据（不解析 PDF）----------
def list_chunk_objects(minio_client: Minio, bucket: str, suffix: str = ".chunks.json") -> list[str]:
    """列出桶中所有「分片数据」对象，默认后缀 .chunks.json。"""
    names = []
    for obj in minio_client.list_objects(bucket, recursive=True):
        name = getattr(obj, "object_name", None) or ""
        if name.endswith("/"):
            continue
        if suffix and name.lower().endswith(suffix.lower()):
            names.append(name)
    return names


def load_chunks_json(minio_client: Minio, bucket: str, object_name: str) -> dict[str, Any] | None:
    """从 MinIO 读取一个分片 JSON 对象，返回 {"doc_name", "doc_path_name", "chunks"} 或 None。"""
    try:
        resp = minio_client.get_object(bucket, object_name)
        data = json.loads(resp.read().decode("utf-8"))
        resp.close()
        resp.release_conn()
    except Exception:
        return None
    chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else None
    if not chunks:
        return None
    if object_name.lower().endswith(".chunks.json"):
        default_doc_path = object_name[: -len(".chunks.json")]
    else:
        default_doc_path = object_name
    doc_path = data.get("doc_path_name") or data.get("doc_name") or default_doc_path
    doc_name = data.get("doc_name") or os.path.basename(doc_path)
    return {"doc_path_name": doc_path, "doc_name": doc_name, "chunks": chunks}


def collect_chunks_from_minio(
    minio_client: Minio,
    bucket: str,
    chunk_object_names: list[str],
    max_docs: int,
    rng: random.Random,
    chunks_suffix: str = ".chunks.json",
) -> list[tuple[str, str, int, str]]:
    """
    从 MinIO 已分片 JSON 中读取，展开为 (doc_path_name, doc_name, chunk_index, chunk_text)。
    不调用任何 PDF 解析器。
    """
    chosen = rng.sample(chunk_object_names, min(max_docs, len(chunk_object_names)))
    flat = []
    for name in chosen:
        doc = load_chunks_json(minio_client, bucket, name)
        if not doc:
            continue
        doc_path = doc["doc_path_name"]
        doc_name = doc["doc_name"]
        for i, content in enumerate(doc["chunks"]):
            if content is None:
                continue
            text = (content if isinstance(content, str) else str(content)).strip()
            if not text:
                continue
            flat.append((doc_path, doc_name, i, text))
    return flat


# ---------- 从 MinIO 列举文档并解析得到 chunks（与 agribot_index 一致）----------
def list_document_objects(minio_client: Minio, bucket: str) -> list[str]:
    """
    与 agribot_index 一致：list_objects(bucket, recursive=True)，跳过目录，
    只保留可解析的文档后缀（.pdf, .docx 等）。
    """
    supported = (".pdf", ".docx", ".doc", ".txt", ".md")
    names = []
    for obj in minio_client.list_objects(bucket, recursive=True):
        name = getattr(obj, "object_name", None) or ""
        if name.endswith("/"):
            continue
        if any(name.lower().endswith(s) for s in supported):
            names.append(name)
    return names


def _parse_doc_chunks(parser: Any, bucket: str, object_name: str) -> dict[str, Any] | None:
    """解析单个 PDF/文档，返回 doc 元数据 + contents；失败返回 None。"""
    try:
        result = parser.parse_object(bucket, object_name)
        if not result or "contents" not in result or not result["contents"]:
            return None
        return result
    except Exception:
        return None


def collect_chunks_by_parsing(
    minio_client: Minio,
    bucket: str,
    object_names: list[str],
    max_docs: int,
    rng: random.Random,
) -> list[tuple[str, str, int, str]]:
    """
    对 object_names 随机取最多 max_docs 个，用 SimpleObjectParser 解析后展开为
    (doc_path_name, doc_name, chunk_index, chunk_text)。仅当 --source parse 时使用。
    """
    from parser.simple_object_parser import SimpleObjectParser  # noqa: E402
    parser = SimpleObjectParser()
    chosen = rng.sample(object_names, min(max_docs, len(object_names)))
    flat = []
    for name in chosen:
        parsed = _parse_doc_chunks(parser, bucket, name)
        if not parsed:
            continue
        doc_path = parsed["doc_path_name"]
        doc_name = parsed.get("doc_name", os.path.basename(name))
        for i, content in enumerate(parsed["contents"]):
            if not (content and content.strip()):
                continue
            flat.append((doc_path, doc_name, i, content.strip()))
    return flat


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
    固定参数版本：
      - source: milvus
      - chunks-suffix: .chunks.json
      - count: 30
      - max-docs: 20
      - context-size: 3
      - seed: None
      - output: /export/workspace/rag/experiment/eval/data/rag_eval_result.json
    如需调整，直接改下面常量即可。
    """
    SOURCE = "milvus"
    CHUNKS_SUFFIX = ".chunks.json"
    COUNT = 30
    MAX_DOCS = 20
    CONTEXT_SIZE = 3
    SEED = None
    OUT_PATH = "/export/workspace/rag/experiment/eval/data/rag_eval_result.json"

    rng = random.Random(SEED)

    minio_client = get_minio_client()
    bucket = get_bucket()
    if not minio_client.bucket_exists(bucket):
        print(f"错误: MinIO 桶 '{bucket}' 不存在")
        sys.exit(1)

    if SOURCE == "milvus":
        _ensure_milvus_env()
        try:
            milvus_api, milvus_field = get_milvus_api()
        except Exception as e:
            print(f"Milvus 初始化失败: {e}")
            print("  请确保 .env 中配置 MILVUS_*（可复制 agribot_index/.env 中的 Milvus 配置），或使用 --source parse 从 MinIO 解析。")
            sys.exit(1)
        print("从 Milvus 读取已索引分片（不解析 PDF）…")
        flat = collect_chunks_from_milvus(minio_client, bucket, milvus_api, milvus_field, MAX_DOCS, rng)
        if not flat:
            print("错误: Milvus 中未找到已索引的文档分片。请先通过 agribot_index 将 MinIO 中的文档解析并写入 Milvus。")
            sys.exit(1)
        print(f"共得到 {len(flat)} 个分片（来自 Milvus），将随机抽取 {args.count} 条生成问题")
    elif SOURCE == "chunks":
        object_names = list_chunk_objects(minio_client, bucket, suffix=CHUNKS_SUFFIX)
        if not object_names:
            print(f"错误: 桶中未找到后缀为 '{args.chunks_suffix}' 的分片数据。")
            print("  请确保 MinIO 中已上传已分片 JSON，或使用 --source milvus 从 Milvus 读取，或 --source parse 从桶内文档现场解析。")
            sys.exit(1)
        print(f"MinIO 中共找到 {len(object_names)} 个分片文件（{CHUNKS_SUFFIX}），将随机选取最多 {MAX_DOCS} 个")
        flat = collect_chunks_from_minio(minio_client, bucket, object_names, MAX_DOCS, rng, CHUNKS_SUFFIX)
    else:
        # --source parse：与 agribot_index 一致，从 MinIO 取文档并解析得分片
        object_names = list_document_objects(minio_client, bucket)
        if not object_names:
            print("错误: 桶中没有找到可解析的文档（.pdf / .docx / .txt / .md）")
            sys.exit(1)
        print(f"MinIO 中共找到 {len(object_names)} 个文档，将随机选取最多 {MAX_DOCS} 个并解析得分片（会调用 PDF/文档解析器）")
        flat = collect_chunks_by_parsing(minio_client, bucket, object_names, MAX_DOCS, rng)

    if not flat:
        print("错误: 未得到任何分片")
        sys.exit(1)
    if args.source != "milvus":
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
    print(f"已写入 {len(results)} 条到 {out_path}")


if __name__ == "__main__":
    main()

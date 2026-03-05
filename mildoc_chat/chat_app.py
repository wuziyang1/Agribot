import logging
import os
import json
from typing import Any, Dict

from flask import Flask, jsonify, request, render_template, Response, stream_with_context

from mildoc_chat.routers.logging_utils import setup_logging
from mildoc_chat.rag.rag_config import Config
from mildoc_chat.rag.rag_service import get_rag_service, RAGResponse


logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")


def _response_to_dict(resp: RAGResponse) -> Dict[str, Any]:
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
            for d in resp.source_documents
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


@app.get("/")
def index():
    """简易 ChatGPT 风格网页"""
    return render_template("chat.html")


@app.get("/health")
def health():
    rag = get_rag_service()
    if rag is None:
        return jsonify({"status": "degraded", "rag_service": "init_failed"})
    try:
        return jsonify(rag.health_check())
    except Exception as e:
        logger.exception("health_check failed")
        return jsonify({"status": "error", "error": str(e)})


@app.post("/api/ask")
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    use_rerank = bool(payload.get("use_rerank", True))
    use_rag = bool(payload.get("use_rag", True))

    if not question:
        return jsonify(
            {
                "success": False,
                "content": "",
                "error_message": "问题内容不能为空",
                "source_documents": [],
                "token_usage": None,
                "scene_info": None,
            }
        )

    rag = get_rag_service()
    if rag is None:
        return jsonify(
            {
                "success": False,
                "content": "",
                "error_message": "RAG 服务初始化失败，请检查 mildoc_chat 的 .env 配置。",
                "source_documents": [],
                "token_usage": None,
                "scene_info": None,
            }
        )

    try:
        resp = rag.query_service(question, use_rerank=use_rerank, use_rag=use_rag)
    except Exception as e:
        logger.exception("query_service failed")
        return jsonify(
            {
                "success": False,
                "content": "",
                "error_message": f"查询过程中发生错误：{e}",
                "source_documents": [],
                "token_usage": None,
                "scene_info": None,
            }
        )

    return jsonify(_response_to_dict(resp))


@app.post("/api/ask_stream")
def api_ask_stream():
    """流式输出回答内容（JSON Lines 格式）

    前端通过 fetch + ReadableStream 逐行解析：
      - type=chunk:   实时追加回答内容
      - type=end:     最终的 RAG 元数据（文档来源、token 使用等）
      - type=error:   出错信息
    """
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    use_rerank = bool(payload.get("use_rerank", True))
    use_rag = bool(payload.get("use_rag", True))

    if not question:
        error_obj = {
            "type": "error",
            "data": {
                "success": False,
                "content": "",
                "error_message": "问题内容不能为空",
                "source_documents": [],
                "token_usage": None,
                "scene_info": None,
            },
        }
        return Response(json.dumps(error_obj) + "\n", mimetype="application/json")

    rag = get_rag_service()
    if rag is None:
        error_obj = {
            "type": "error",
            "data": {
                "success": False,
                "content": "",
                "error_message": "RAG 服务初始化失败，请检查 mildoc_chat 的 .env 配置。",
                "source_documents": [],
                "token_usage": None,
                "scene_info": None,
            },
        }
        return Response(json.dumps(error_obj) + "\n", mimetype="application/json")

    def generate():
        try:
            # 先返回一个开始事件，前端可以据此清空旧的引用信息等
            start_obj = {"type": "start"}
            yield json.dumps(start_obj, ensure_ascii=False) + "\n"

            resp = rag.query_service(question, use_rerank=use_rerank, use_rag=use_rag)
        except Exception as e:
            logger.exception("query_service failed (stream)")
            err_obj = {
                "type": "error",
                "data": {
                    "success": False,
                    "content": "",
                    "error_message": f"查询过程中发生错误：{e}",
                    "source_documents": [],
                    "token_usage": None,
                    "scene_info": None,
                },
            }
            yield json.dumps(err_obj, ensure_ascii=False) + "\n"
            return

        # 如果业务失败，同样只发一条 error
        if not resp.success:
            err_obj = {"type": "error", "data": _response_to_dict(resp)}
            yield json.dumps(err_obj, ensure_ascii=False) + "\n"
            return

        answer = resp.content or ""
        chunk_size = 60  # 以字符数简单切分，主要用于前端渐进展示

        for i in range(0, len(answer), chunk_size):
            piece = answer[i : i + chunk_size]
            chunk_obj = {"type": "chunk", "data": {"content": piece}}
            yield json.dumps(chunk_obj, ensure_ascii=False) + "\n"

        # 最后一条带有 RAG 元数据，前端可用于展示引用文档、token 用量等
        end_obj = {"type": "end", "data": _response_to_dict(resp)}
        yield json.dumps(end_obj, ensure_ascii=False) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/json")


if __name__ == "__main__":
    log_path = os.getenv("MILDOC_CHAT_LOG", os.path.join(os.path.dirname(__file__), "mildoc_chat.log"))
    setup_logging(log_path=log_path, level=logging.INFO)

    host = os.getenv("MILDOC_CHAT_HOST", "0.0.0.0")
    port = int(os.getenv("MILDOC_CHAT_PORT", "8890"))
    debug = os.getenv("MILDOC_CHAT_DEBUG", "False").lower() == "true"

    logger.info("Mildoc Chat Starting...")
    logger.info(f"listen: http://{host}:{port}")
    logger.info(f"milvus: {Config.MILVUS_HOST}:{Config.MILVUS_PORT} db={Config.MILVUS_DATABASE} col={Config.MILVUS_COLLECTION_NAME}")

    app.run(host=host, port=port, debug=debug, threaded=True)


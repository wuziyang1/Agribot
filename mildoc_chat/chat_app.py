import logging
import os
import json
from typing import Any, Dict

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)

from mildoc_chat.routers.logging_utils import setup_logging
from mildoc_chat.rag.rag_config import Config
from mildoc_chat.rag.rag_service import get_rag_service, RAGResponse
from mildoc_chat.routers.login_flask import login_bp
from mildoc_chat.routers.register_flask import register_bp
from mildoc_chat.forgot_password_smtp import send_reset_code, reset_password


logger = logging.getLogger(__name__)

# 使用 app 目录下的模板和静态资源
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# session 加密密钥（生产环境请通过环境变量覆盖）
app.secret_key = os.getenv("MILDOC_CHAT_SECRET_KEY", "change-me-in-production")

# 注册登录 / 注册蓝图
app.register_blueprint(login_bp)
app.register_blueprint(register_bp)


@app.before_request
def _require_login():
    """全局登录校验：未登录时先进入登录页，再访问聊天界面或接口。"""
    exempt_endpoints = {
        "health",
        "mildoc_chat_login.login",
        "mildoc_chat_login.login_post",
        "mildoc_chat_login.logout",
        "mildoc_chat_register.register",
        "mildoc_chat_register.register_post",
        "mildoc_chat_register.api_register_send_code",
        "forgot_password",          # 找回密码页本身
        "api_forgot_send_code",     # 找回密码：发送验证码
        "api_forgot_reset",         # 找回密码：重置密码
        "static",
    }

    # 某些情况 request.endpoint 可能为 None（如 404），直接放行
    if not request.endpoint or request.endpoint in exempt_endpoints:
        return

    if "chat_username" in session:
        return

    # 未登录访问 API：返回 JSON 提示
    if request.path.startswith("/api/"):
        return (
            jsonify(
                {
                    "success": False,
                    "content": "",
                    "error_message": "请先登录后再使用聊天功能",
                    "source_documents": [],
                    "token_usage": None,
                    "scene_info": None,
                }
            ),
            401,
        )

    # 其它页面：跳转到登录页
    return redirect(url_for("mildoc_chat_login.login"))


@app.get("/forgot")
def forgot_password():
    """找回密码页面：通过邮箱验证码重置密码。"""
    return render_template("forgot_password.html")


@app.post("/api/forgot/send_code")
def api_forgot_send_code():
    """找回密码：发送邮箱验证码。"""
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip()
    result = send_reset_code(email)
    return jsonify(result)


@app.post("/api/forgot/reset")
def api_forgot_reset():
    """找回密码：提交验证码 + 新密码，重置登录密码。"""
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip()
    code = (payload.get("verification_code") or "").strip()
    new_password = (payload.get("new_password") or "").strip()
    result = reset_password(email, code, new_password)
    return jsonify(result)


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

    def generate():
        rag = get_rag_service()
        if rag is None:
            err_obj = {
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
            yield json.dumps(err_obj, ensure_ascii=False) + "\n"
            return

        try:
            # 直接使用 RAGService 自带的 token 级流式接口
            for event in rag.stream_query(question, use_rerank=use_rerank, use_rag=use_rag):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("stream_query failed")
            err_resp = RAGResponse(
                content="",
                source_documents=[],
                success=False,
                error_message=f"查询过程中发生错误：{e}",
                scene_info=None,
            )
            err_obj = {"type": "error", "data": _response_to_dict(err_resp)}
            yield json.dumps(err_obj, ensure_ascii=False) + "\n"

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


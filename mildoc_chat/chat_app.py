import logging
import os
import json
import re
import secrets
import time
import smtplib
from functools import wraps
from typing import Any, Dict, Optional, Tuple

from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    Response,
    stream_with_context,
    session,
    redirect,
    url_for,
)

from mildoc_chat.routers.database import (
    ensure_users_table_exists,
    get_user_by_username,
    get_user_by_email,
    create_user,
    update_user_password_by_username,
)
from mildoc_chat.routers.logging_utils import setup_logging
from mildoc_chat.routers.login_flask import login_bp
from mildoc_chat.routers.register_flask import (
    register_bp,
    send_email_code as _send_email_code,
    validate_email as _validate_email,
    validate_password as _validate_password,
    validate_username as _validate_username,
)
from mildoc_chat.rag.rag_config import Config
from mildoc_chat.rag.rag_service import get_rag_service, RAGResponse


logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# 对 session 加密
app.secret_key = os.getenv("MILDOC_CHAT_SECRET_KEY", "mildoc-chat-secret")

# 注册 Blueprint（登录 & 注册）
app.register_blueprint(login_bp)
app.register_blueprint(register_bp)

# 兼容旧的环境变量账号（可选）—— 仅用于找回密码逻辑
CHAT_USERNAME = os.getenv("CHAT_USERNAME")
CHAT_PASSWORD = os.getenv("CHAT_PASSWORD")

# 初始管理员邮箱（用于环境变量 admin 用户的找回密码）
ADMIN_INITIAL_EMAIL = os.getenv("MILDOC_CHAT_ADMIN_EMAIL", "3442557641@qq.com").strip()

# SMTP（可选：未配置则仅在日志里输出验证码）
# 优先使用通用 SMTP_* 变量，兼容 MILDOC_CHAT_SMTP_* 前缀
SMTP_HOST = (os.getenv("SMTP_SERVER") or os.getenv("MILDOC_CHAT_SMTP_HOST") or "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT") or os.getenv("MILDOC_CHAT_SMTP_PORT") or "587")
SMTP_USER = (os.getenv("SMTP_USERNAME") or os.getenv("MILDOC_CHAT_SMTP_USER") or "").strip()
SMTP_PASSWORD = (os.getenv("SMTP_PASSWORD") or os.getenv("MILDOC_CHAT_SMTP_PASSWORD") or "").strip()
SMTP_FROM = (
    os.getenv("FROM_EMAIL")
    or os.getenv("MILDOC_CHAT_SMTP_FROM")
    or SMTP_USER
).strip()

# 忘记密码验证码缓存（内存）
_forgot_codes: Dict[str, Dict[str, Any]] = {}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "chat_username" not in session:
            # 指向登录 Blueprint 的 login 视图
            return redirect(url_for("mildoc_chat_login.login"))
        return f(*args, **kwargs)

    return decorated


def _ensure_db_ready() -> None:
    try:
        ensure_users_table_exists()
    except Exception:  # noqa: BLE001
        logger.exception("ensure users table failed")


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
@login_required
def index():
    """简易 ChatGPT 风格网页"""
    return render_template("chat.html")


@app.get("/forgot_password")
def forgot_password():
    if "chat_username" in session:
        return redirect(url_for("index"))
    return render_template("forgot_password.html")


@app.post("/api/forgot/send_code")
def api_forgot_send_code():
    """发送找回密码验证码（用户名 + 邮箱校验）"""
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    email = (payload.get("email") or "").strip()

    err = _validate_username(username)
    if err:
        return jsonify({"success": False, "error_message": err})
    err = _validate_email(email)
    if err:
        return jsonify({"success": False, "error_message": err})

    user = None
    try:
        user = get_user_by_username(username)
    except Exception:  # noqa: BLE001
        logger.exception("db get_user_by_username failed (forgot send_code)")
        return jsonify({"success": False, "error_message": "数据库连接失败，请检查本地 MySQL 配置"})

    # 绑定邮箱：优先使用 users.json 中的邮箱；若用户不存在但匹配环境变量 admin，则使用初始管理员邮箱
    bound_email: Optional[str] = None
    if user and (user.get("email") or "").strip():
        bound_email = (user.get("email") or "").strip()
    elif CHAT_USERNAME and username == CHAT_USERNAME and ADMIN_INITIAL_EMAIL:
        bound_email = ADMIN_INITIAL_EMAIL

    if not bound_email:
        return jsonify({"success": False, "error_message": "该用户未绑定邮箱或不存在"})

    key_email = bound_email.strip().lower()
    if key_email != email.strip().lower():
        return jsonify({"success": False, "error_message": "用户名与邮箱不匹配"})

    now = time.time()
    rec = _forgot_codes.get(key_email) or {}
    last_sent = float(rec.get("sent_at") or 0)
    cooldown = 60
    if now - last_sent < cooldown:
        return jsonify({"success": True, "cooldown_seconds": int(cooldown - (now - last_sent))})

    code = f"{secrets.randbelow(1000000):06d}"
    ok, msg = _send_email_code(email, code, scene="forgot")
    if not ok:
        return jsonify({"success": False, "error_message": msg})

    _forgot_codes[key_email] = {
        "code": code,
        "username": username,
        "sent_at": now,
        "expires_at": now + 600,
    }
    return jsonify({"success": True, "cooldown_seconds": cooldown})


@app.post("/api/forgot/reset")
def api_forgot_reset():
    """通过邮箱验证码重置密码"""
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    email = (payload.get("email") or "").strip()
    code = (payload.get("verification_code") or "").strip()
    new_password = payload.get("new_password") or ""

    err = _validate_username(username)
    if err:
        return jsonify({"success": False, "error_message": err})
    err = _validate_email(email)
    if err:
        return jsonify({"success": False, "error_message": err})
    err = _validate_password(new_password)
    if err:
        return jsonify({"success": False, "error_message": err})

    user = None
    try:
        user = get_user_by_username(username)
    except Exception:  # noqa: BLE001
        logger.exception("db get_user_by_username failed (forgot reset)")
        return jsonify({"success": False, "error_message": "数据库连接失败，请检查本地 MySQL 配置"})

    bound_email: Optional[str] = None
    if user and (user.get("email") or "").strip():
        bound_email = (user.get("email") or "").strip()
    elif CHAT_USERNAME and username == CHAT_USERNAME and ADMIN_INITIAL_EMAIL:
        bound_email = ADMIN_INITIAL_EMAIL

    if not bound_email:
        return jsonify({"success": False, "error_message": "该用户未绑定邮箱或不存在"})

    key_email = bound_email.strip().lower()
    if key_email != email.strip().lower():
        return jsonify({"success": False, "error_message": "用户名与邮箱不匹配"})

    rec = _forgot_codes.get(key_email)
    now = time.time()
    if not rec or rec.get("expires_at", 0) < now:
        return jsonify({"success": False, "error_message": "验证码不存在或已过期，请重新获取"})
    if rec.get("username") != username:
        return jsonify({"success": False, "error_message": "验证码与用户名不匹配"})
    if str(rec.get("code") or "") != code:
        return jsonify({"success": False, "error_message": "验证码错误"})

    from werkzeug.security import generate_password_hash

    try:
        hashed = generate_password_hash(new_password)
        if not user and CHAT_USERNAME and username == CHAT_USERNAME:
            create_user(username=username, email=bound_email, hashed_password=hashed)
        else:
            ok = update_user_password_by_username(username=username, hashed_password=hashed)
            if not ok:
                return jsonify({"success": False, "error_message": "密码更新失败：用户不存在或已被删除"})
    except Exception as e:  # noqa: BLE001
        logger.exception("reset password db failed")
        return jsonify({"success": False, "error_message": f"密码更新失败：{e}"})

    _forgot_codes.pop(key_email, None)
    return jsonify({"success": True})


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
@login_required
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    use_rerank = bool(payload.get("use_rerank", True))

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
        resp = rag.query_service(question, use_rerank=use_rerank)
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
@login_required
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

            resp = rag.query_service(question, use_rerank=use_rerank)
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
    log_path = os.getenv(
        "MILDOC_CHAT_LOG",
        os.path.join(os.path.dirname(__file__), "routers", "mildoc_chat.log"),
    )
    setup_logging(log_path=log_path, level=logging.INFO)
    _ensure_db_ready()

    host = os.getenv("MILDOC_CHAT_HOST", "0.0.0.0")
    port = int(os.getenv("MILDOC_CHAT_PORT", "8890"))
    debug = os.getenv("MILDOC_CHAT_DEBUG", "False").lower() == "true"

    logger.info("Mildoc Chat Starting...")
    logger.info(f"listen: http://{host}:{port}")
    logger.info(f"milvus: {Config.MILVUS_HOST}:{Config.MILVUS_PORT} db={Config.MILVUS_DATABASE} col={Config.MILVUS_COLLECTION_NAME}")

    app.run(host=host, port=port, debug=debug, threaded=True)


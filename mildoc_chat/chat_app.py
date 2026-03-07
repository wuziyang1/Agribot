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
from mildoc_chat.rag.graph_rag_service import get_graph_rag_service
from mildoc_chat.routers.login_flask import login_bp
from mildoc_chat.routers.register_flask import register_bp
from mildoc_chat.routers.database import (
    create_message,
    create_session,
    ensure_messages_table_exists,
    ensure_sessions_table_exists,
    ensure_users_table_exists,
    get_user_by_username,
    list_messages,
    list_sessions,
    set_active_session,
    delete_session,
    update_session_title,
    update_user_profile,
)
from mildoc_chat.forgot_password_smtp import send_reset_code, reset_password


logger = logging.getLogger(__name__)

# 使用 app 目录下的模板和静态资源
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# session 加密密钥（生产环境请通过环境变量覆盖）
app.secret_key = os.getenv("MILDOC_CHAT_SECRET_KEY", "change-me-in-production")

# 注册登录 / 注册蓝图
app.register_blueprint(login_bp)
app.register_blueprint(register_bp)

try:
    # 确保核心表存在（users 由注册/登录依赖，sessions/messages 用于会话持久化）
    ensure_users_table_exists()
    ensure_sessions_table_exists()
    ensure_messages_table_exists()
except Exception:  # noqa: BLE001
    logger.exception("ensure tables failed")


def _current_user():
    username = session.get("chat_username")
    if not username:
        return None, None
    user = get_user_by_username(username)
    return username, user


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
    }


@app.get("/")
def index():
    """简易 ChatGPT 风格网页"""
    username = session.get("chat_username") or "Mildoc 用户"
    return render_template("chat.html", username=username)


@app.get("/profile")
def profile():
    """个人主页"""
    username = session.get("chat_username")
    if not username:
        return redirect(url_for("mildoc_chat_login.login"))
    user = get_user_by_username(username)
    if not user:
        return redirect(url_for("index"))
    created_at = user.get("created_at")
    created_str = created_at.strftime("%Y-%m-%d %H:%M") if hasattr(created_at, "strftime") and created_at else "—"
    return render_template(
        "profile.html",
        username=user.get("username") or username,
        email=user.get("email") or "",
        created_at=created_str,
    )


@app.post("/api/profile/update")
def api_profile_update():
    """更新个人资料（用户名、邮箱）"""
    username = session.get("chat_username")
    if not username:
        return jsonify({"success": False, "error_message": "请先登录"}), 401
    user = get_user_by_username(username)
    if not user:
        return jsonify({"success": False, "error_message": "用户不存在"}), 404
    payload = request.get_json(silent=True) or {}
    new_username = (payload.get("username") or "").strip() or None if "username" in payload else None  # noqa: E501
    new_email = (payload.get("email") or "").strip() or None if "email" in payload else None  # noqa: E501
    ok, err = update_user_profile(user_id=user["id"], username=new_username, email=new_email)
    if not ok:
        return jsonify({"success": False, "error_message": err}), 400
    if new_username:
        session["chat_username"] = new_username
    return jsonify({"success": True})


@app.get("/api/sessions")
def api_sessions_list():
    _, user = _current_user()
    if not user:
        return jsonify({"success": False, "error_message": "请先登录"}), 401
    items = list_sessions(user_id=int(user["id"]))
    return jsonify({"success": True, "sessions": items})


@app.post("/api/sessions")
def api_sessions_create():
    _, user = _current_user()
    if not user:
        return jsonify({"success": False, "error_message": "请先登录"}), 401
    sess = create_session(user_id=int(user["id"]))
    return jsonify({"success": True, "session": sess})


@app.patch("/api/sessions/<session_id>")
def api_sessions_update(session_id: str):
    _, user = _current_user()
    if not user:
        return jsonify({"success": False, "error_message": "请先登录"}), 401
    payload = request.get_json(silent=True) or {}
    if "is_active" in payload and bool(payload.get("is_active")):
        ok = set_active_session(user_id=int(user["id"]), session_id=session_id)
        return jsonify({"success": ok})
    if "title" in payload:
        ok = update_session_title(user_id=int(user["id"]), session_id=session_id, title=payload.get("title"))
        return jsonify({"success": ok})
    return jsonify({"success": True})


@app.delete("/api/sessions/<session_id>")
def api_sessions_delete(session_id: str):
    _, user = _current_user()
    if not user:
        return jsonify({"success": False, "error_message": "请先登录"}), 401
    ok = delete_session(user_id=int(user["id"]), session_id=session_id)
    return jsonify({"success": ok})


@app.get("/api/sessions/<session_id>/messages")
def api_messages_list(session_id: str):
    _, user = _current_user()
    if not user:
        return jsonify({"success": False, "error_message": "请先登录"}), 401
    items = list_messages(user_id=int(user["id"]), session_id=session_id)
    return jsonify({"success": True, "messages": items})


@app.post("/api/sessions/<session_id>/messages")
def api_messages_create(session_id: str):
    _, user = _current_user()
    if not user:
        return jsonify({"success": False, "error_message": "请先登录"}), 401
    payload = request.get_json(silent=True) or {}
    role = (payload.get("role") or "").strip()
    content = payload.get("content") or ""
    try:
        ok = create_message(user_id=int(user["id"]), session_id=session_id, role=role, content=content)
    except ValueError:
        return jsonify({"success": False, "error_message": "role 非法"}), 400
    return jsonify({"success": ok})


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
    session_id = (payload.get("session_id") or "").strip() or None

    chat_history = []
    if session_id:
        _, user = _current_user()
        if user:
            try:
                rows = list_messages(user_id=int(user["id"]), session_id=session_id)
                for m in rows:
                    role = (m.get("role") or "").strip().lower()
                    content = (m.get("content") or "").strip()
                    if role in ("user", "assistant", "system") and content:
                        chat_history.append({"role": role if role != "system" else "assistant", "content": content})
            except Exception:  # noqa: BLE001
                pass

    if not question:
        return jsonify(
            {
                "success": False,
                "content": "",
                "error_message": "问题内容不能为空",
                "source_documents": [],
                "token_usage": None,
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
            }
        )

    try:
        resp = rag.query_service(question, use_rerank=use_rerank, use_rag=use_rag, chat_history=chat_history)
    except Exception as e:
        logger.exception("query_service failed")
        return jsonify(
            {
                "success": False,
                "content": "",
                "error_message": f"查询过程中发生错误：{e}",
                "source_documents": [],
                "token_usage": None,
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
    session_id = (payload.get("session_id") or "").strip() or None

    # 若提供 session_id 且已登录，则加载该会话的历史消息供模型参考
    chat_history: list = []
    if session_id:
        _, user = _current_user()
        if user:
            try:
                rows = list_messages(user_id=int(user["id"]), session_id=session_id)
                for m in rows:
                    role = (m.get("role") or "").strip().lower()
                    content = (m.get("content") or "").strip()
                    if role in ("user", "assistant", "system") and content:
                        chat_history.append({"role": role if role != "system" else "assistant", "content": content})
            except Exception:  # noqa: BLE001
                pass

    if not question:
        error_obj = {
            "type": "error",
            "data": {
                "success": False,
                "content": "",
                "error_message": "问题内容不能为空",
                "source_documents": [],
                "token_usage": None,
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
                },
            }
            yield json.dumps(err_obj, ensure_ascii=False) + "\n"
            return

        try:
            # 直接使用 RAGService 自带的 token 级流式接口（传入会话历史）
            for event in rag.stream_query(question, use_rerank=use_rerank, use_rag=use_rag, chat_history=chat_history):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("stream_query failed")
            err_resp = RAGResponse(
                content="",
                source_documents=[],
                success=False,
                error_message=f"查询过程中发生错误：{e}",
            )
            err_obj = {"type": "error", "data": _response_to_dict(err_resp)}
            yield json.dumps(err_obj, ensure_ascii=False) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/json")



# =========================================================================
# Graph RAG API 路由
# =========================================================================

@app.post("/api/graph/ask_stream")
def api_graph_ask_stream():
    """Graph RAG 流式问答（知识图谱检索）

    前端格式与 /api/ask_stream 完全一致（JSON Lines）。
    """
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    session_id = (payload.get("session_id") or "").strip() or None

    chat_history: list = []
    if session_id:
        _, user = _current_user()
        if user:
            try:
                rows = list_messages(user_id=int(user["id"]), session_id=session_id)
                for m in rows:
                    role = (m.get("role") or "").strip().lower()
                    content = (m.get("content") or "").strip()
                    if role in ("user", "assistant", "system") and content:
                        chat_history.append({"role": role if role != "system" else "assistant", "content": content})
            except Exception:
                pass

    if not question:
        error_obj = {
            "type": "error",
            "data": {
                "success": False,
                "content": "",
                "error_message": "问题内容不能为空",
                "source_documents": [],
                "token_usage": None,
            },
        }
        return Response(json.dumps(error_obj) + "\n", mimetype="application/json")

    def generate():
        graph_rag = get_graph_rag_service()
        if graph_rag is None:
            err_obj = {
                "type": "error",
                "data": {
                    "success": False,
                    "content": "",
                    "error_message": "Graph RAG 服务不可用，请检查 Neo4j 配置。",
                    "source_documents": [],
                    "token_usage": None,
                },
            }
            yield json.dumps(err_obj, ensure_ascii=False) + "\n"
            return

        try:
            for event in graph_rag.stream_query(question, chat_history=chat_history):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.exception("graph stream_query failed")
            err_obj = {
                "type": "error",
                "data": {
                    "success": False,
                    "content": "",
                    "error_message": f"图谱查询失败：{e}",
                    "source_documents": [],
                    "token_usage": None,
                },
            }
            yield json.dumps(err_obj, ensure_ascii=False) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/json")


@app.post("/api/graph/import")
def api_graph_import():
    """导入文本到知识图谱"""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    doc_name = (payload.get("doc_name") or "").strip() or "手动导入"

    if not text:
        return jsonify({"success": False, "error_message": "文本内容不能为空"}), 400

    graph_rag = get_graph_rag_service()
    if graph_rag is None:
        return jsonify({"success": False, "error_message": "Graph RAG 服务不可用"}), 503

    result = graph_rag.import_text(text, doc_name=doc_name)
    return jsonify({
        "success": result.success,
        "entities_count": result.entities_count,
        "relations_count": result.relations_count,
        "chunks_processed": result.chunks_processed,
        "error_message": result.error_message,
    })


@app.get("/api/graph/stats")
def api_graph_stats():
    """获取知识图谱统计信息"""
    graph_rag = get_graph_rag_service()
    if graph_rag is None:
        return jsonify({"success": False, "error_message": "Graph RAG 服务不可用"}), 503
    stats = graph_rag.get_stats()
    return jsonify({"success": True, **stats})


@app.post("/api/graph/clear")
def api_graph_clear():
    """清空知识图谱"""
    graph_rag = get_graph_rag_service()
    if graph_rag is None:
        return jsonify({"success": False, "error_message": "Graph RAG 服务不可用"}), 503
    ok = graph_rag.clear_graph()
    return jsonify({"success": ok})


@app.get("/api/graph/health")
def api_graph_health():
    """Graph RAG 健康检查"""
    graph_rag = get_graph_rag_service()
    if graph_rag is None:
        return jsonify({"status": "unavailable", "service": "GraphRAGService"})
    return jsonify(graph_rag.health_check())


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


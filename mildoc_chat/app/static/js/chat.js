const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const sendLabel = document.getElementById('send-label');
const statusLabel = document.getElementById('status-label');
const statusBadge = document.querySelector('.badge');
const hintText = document.getElementById('hint-text');
const ragToggleBtn = document.getElementById('rag-toggle-btn');
const ragToggleLabel = document.getElementById('rag-toggle-label');
const navSessionsList = document.getElementById('nav-sessions-list');
const navNewChatBtn = document.querySelector('.nav-new-chat');

/** 是否使用知识库（RAG）检索，默认开启 */
let useRag = true;

/** 始终使用知识图谱（Graph RAG）检索 */
const useGraph = true;

/** Session 管理（数据库持久化） */
const WELCOME_MSG = '你好，我是 Mildoc Chat，可以基于 Milvus + 文档知识库回答你的问题，并在下方展示引用的文档来源。';

const sessions = [];
let activeSessionId = null;
const messagesCache = new Map(); // session_id -> [{role, content}]

async function apiJson(url, options) {
  const resp = await fetch(url, options || {});
  let data = null;
  try {
    data = await resp.json();
  } catch (e) {
    data = null;
  }
  if (!resp.ok) {
    const msg = (data && data.error_message) || '请求失败';
    throw new Error(msg);
  }
  return data;
}

function getActiveSession() {
  return sessions.find((s) => s.id === activeSessionId);
}

async function loadSessions() {
  const data = await apiJson('/api/sessions');
  sessions.length = 0;
  (data.sessions || []).forEach((s) => {
    sessions.push({
      id: s.session_id,
      name: s.title || null,
      is_active: !!s.is_active
    });
  });
  const active = sessions.find((s) => s.is_active) || sessions[0] || null;
  activeSessionId = active ? active.id : null;
  renderSessionsList();
}

async function createNewSession() {
  const data = await apiJson('/api/sessions', { method: 'POST' });
  const s = data.session;
  await loadSessions();
  return { id: s.session_id };
}

async function loadMessages(sessionId) {
  const data = await apiJson('/api/sessions/' + encodeURIComponent(sessionId) + '/messages');
  const msgs = (data.messages || []).map((m) => ({
    role: m.role === 'assistant' ? 'bot' : m.role === 'system' ? 'bot' : 'user',
    content: m.content || ''
  }));
  messagesCache.set(sessionId, msgs.length ? msgs : [{ role: 'bot', content: WELCOME_MSG }]);
  return messagesCache.get(sessionId);
}

async function switchToSession(sessionId) {
  activeSessionId = sessionId;
  await apiJson('/api/sessions/' + encodeURIComponent(sessionId), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ is_active: true })
  });
  await loadSessions();
  const msgs = await loadMessages(sessionId);
  messagesEl.innerHTML = '';
  msgs.forEach((m) => renderMessage(m.role, m.content));
  messagesEl.scrollTop = messagesEl.scrollHeight;
  renderSessionsList();
}

function renderMessage(role, content) {
  const { row } = createMessageElements(role, content);
  messagesEl.appendChild(row);
}

function appendMessage(role, content) {
  const sid = activeSessionId;
  if (sid) {
    const list = messagesCache.get(sid) || [];
    list.push({ role, content });
    messagesCache.set(sid, list);
  }
  renderMessage(role, content);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderSessionsList() {
  if (!navSessionsList) return;
  navSessionsList.innerHTML = '';
  sessions.forEach((s) => {
    const item = document.createElement('div');
    item.className = 'nav-session-item' + (s.id === activeSessionId ? ' active' : '');
    item.dataset.sessionId = s.id;

    const mainBtn = document.createElement('button');
    mainBtn.type = 'button';
    mainBtn.className = 'nav-session-main';
    mainBtn.textContent = s.name || '新聊天';

    const menuBtn = document.createElement('button');
    menuBtn.type = 'button';
    menuBtn.className = 'nav-session-menu-btn';
    menuBtn.setAttribute('aria-label', '会话菜单');
    menuBtn.textContent = '⋯';

    const menu = document.createElement('div');
    menu.className = 'nav-session-menu';
    menu.innerHTML =
      '<button type="button" class="nav-session-menu-item" data-action="rename">重命名</button>' +
      '<button type="button" class="nav-session-menu-item danger" data-action="delete">删除会话</button>';

    menuBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      closeAllSessionMenus();
      item.classList.toggle('menu-open');
    });

    item.appendChild(mainBtn);
    item.appendChild(menuBtn);
    item.appendChild(menu);
    navSessionsList.appendChild(item);
  });
}

function closeAllSessionMenus() {
  if (!navSessionsList) return;
  navSessionsList.querySelectorAll('.nav-session-item.menu-open').forEach((el) => {
    el.classList.remove('menu-open');
  });
}

function typesetMath(targetEl) {
  if (!targetEl) return;
  if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
    window.MathJax.typesetPromise([targetEl]).catch(function (err) {
      console.error('MathJax 渲染出错:', err);
    });
  }
}

function getMarkedRenderer() {
  if (!window.marked) return null;
  if (getMarkedRenderer._renderer) return getMarkedRenderer._renderer;
  const renderer = new window.marked.Renderer();
  renderer.html = function () {
    return '';
  };
  getMarkedRenderer._renderer = renderer;
  return renderer;
}

function renderBotMarkdown(targetEl, text, opts) {
  if (!targetEl) return;
  const options = opts || {};
  const safeText = text || '';

  if (window.marked && typeof window.marked.parse === 'function') {
    const renderer = getMarkedRenderer();
    targetEl.innerHTML = window.marked.parse(safeText, {
      breaks: true,
      gfm: true,
      renderer: renderer || undefined
    });
  } else {
    targetEl.textContent = safeText;
  }

  if (options.typesetMath) typesetMath(targetEl);
}

function createMessageElements(role, content) {
  const row = document.createElement('div');
  row.className = 'msg-row ' + (role === 'user' ? 'user' : 'bot');

  const avatar = document.createElement('div');
  avatar.className = 'avatar ' + (role === 'user' ? 'user' : 'bot');
  avatar.textContent = role === 'user' ? '我' : 'AI';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const name = document.createElement('div');
  name.className = 'msg-name';
  name.textContent = role === 'user' ? '我' : 'Mildoc Chat';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble ' + (role === 'user' ? 'user' : 'bot');
  if (role === 'user') {
    bubble.textContent = content;
  } else {
    renderBotMarkdown(bubble, content, { typesetMath: true });
  }

  body.appendChild(name);
  body.appendChild(bubble);

  if (role === 'bot') {
    const insightsContainer = document.createElement('div');
    insightsContainer.className = 'msg-insights';
    body.appendChild(insightsContainer);
  }

  if (role === 'user') {
    row.appendChild(body);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(body);
  }

  return { row, bubble };
}

function renderInsightsInto(containerEl, data) {
  if (!containerEl) return;

  const raw = Array.isArray(data.source_documents) ? data.source_documents : [];
  const seen = new Set();
  const docs = raw.filter((d) => {
    const key = d.doc_name || d.doc_path_name || '未命名文档';
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  if (docs.length === 0) {
    containerEl.innerHTML = '';
    return;
  }

  let html = '';
  html += '<div class="insights-section">';
  html += '<span class="insights-title">引用文档</span>';
  html += '<div class="insights-doc-list">';

  docs.slice(0, 4).forEach((d, index) => {
    const name = d.doc_name || d.doc_path_name || '未命名文档';
    const titleAttr = name.replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    html +=
      '<button class="doc-chip" type="button" title="' + titleAttr + '">' +
      '<span class="doc-chip-index">' +
      (index + 1) +
      '</span>' +
      '<span class="doc-chip-name">' +
      name +
      '</span>' +
      '</button>';
  });

  html += '</div></div>';
  containerEl.innerHTML = html;
}

function setLoading(loading) {
  if (loading) {
    sendBtn.disabled = true;
    sendBtn.classList.add('loading');
    statusLabel.textContent = '正在生成回答…';
    if (hintText) {
      hintText.textContent = '正在处理你的问题，请稍候…';
    }
  } else {
    sendBtn.disabled = false;
    sendBtn.classList.remove('loading');
    if (statusLabel) {
      // 根据是否启用知识库展示不同提示
      statusLabel.textContent = useRag ? 'RAG 服务已连接' : '知识库未使用';
    }
    if (hintText) {
      hintText.textContent = '按 Enter 发送，Shift+Enter 换行。';
    }
  }
}

async function sendQuestion() {
  const text = (inputEl.value || '').trim();
  if (!text) {
    if (hintText) {
      hintText.textContent = '请先输入问题。';
    }
    inputEl.focus();
    return;
  }

  const sid = activeSessionId;
  if (!sid) return;
  const cached = messagesCache.get(sid) || [];
  const isFirstUserMessage = !cached.some((m) => m.role === 'user');
  const sess = getActiveSession();
  if (sess && isFirstUserMessage && !sess.name) {
    const title = text.length > 30 ? text.slice(0, 30) + '…' : text;
    sess.name = title;
    renderSessionsList();
    apiJson('/api/sessions/' + encodeURIComponent(sid), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: title })
    }).then(loadSessions).catch(() => {});
  }

  appendMessage('user', text);
  apiJson('/api/sessions/' + encodeURIComponent(sid) + '/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role: 'user', content: text })
  }).catch(() => {});
  inputEl.value = '';
  setLoading(true);

  const { row: botRow, bubble: botBubble } = createMessageElements('bot', '');
  botBubble.classList.add('streaming');
  botBubble.textContent = '';
  messagesEl.appendChild(botRow);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  let accumulated = '';

  try {
    const apiUrl = '/api/ask_stream';
    const resp = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: text,
        use_rerank: true,
        use_rag: useRag,
        use_graph: useGraph,
        session_id: activeSessionId || undefined
      })
    });

    if (!resp.ok || !resp.body) {
      throw new Error('网络请求失败');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let index;
      while ((index = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);

        if (!line) continue;

        let msg;
        try {
          msg = JSON.parse(line);
        } catch (e) {
          console.warn('解析流式响应失败:', e, line);
          continue;
        }

        if (msg.type === 'chunk' && msg.data && typeof msg.data.content === 'string') {
          accumulated += msg.data.content;
          botBubble.textContent = accumulated;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (msg.type === 'end' && msg.data) {
          botBubble.classList.remove('streaming');
          renderBotMarkdown(botBubble, accumulated, { typesetMath: true });
          renderInsightsInto(botRow.querySelector('.msg-insights'), msg.data);
          // 写入缓存 + 数据库
          const list = messagesCache.get(sid) || [];
          list.push({ role: 'bot', content: accumulated });
          messagesCache.set(sid, list);
          apiJson('/api/sessions/' + encodeURIComponent(sid) + '/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ role: 'assistant', content: accumulated })
          }).catch(() => {});
        } else if (msg.type === 'error' && msg.data) {
          const errMsg = msg.data.error_message || '查询失败，请查看后端日志。';
          const errContent = '错误：' + errMsg;
          botBubble.textContent = errContent;
          statusLabel.textContent = '后端返回错误';
          renderInsightsInto(botRow.querySelector('.msg-insights'), { source_documents: [], token_usage: null });
          const list = messagesCache.get(sid) || [];
          list.push({ role: 'bot', content: errContent });
          messagesCache.set(sid, list);
          return;
        }
      }
    }

    if (!accumulated) {
      botBubble.classList.remove('streaming');
      const emptyContent = '（后端未返回内容）';
      renderBotMarkdown(botBubble, emptyContent, { typesetMath: false });
      const list = messagesCache.get(sid) || [];
      list.push({ role: 'bot', content: emptyContent });
      messagesCache.set(sid, list);
    }
  } catch (err) {
    console.error(err);
    botBubble.classList.remove('streaming');
    const errContent = '请求异常，请检查服务是否正常运行。';
    renderBotMarkdown(botBubble, errContent, { typesetMath: false });
    statusLabel.textContent = '请求异常';
    renderInsightsInto(botRow.querySelector('.msg-insights'), { source_documents: [], token_usage: null });
    const list = messagesCache.get(sid) || [];
    list.push({ role: 'bot', content: errContent });
    messagesCache.set(sid, list);
  } finally {
    setLoading(false);
  }
}

function updateRagToggleUI() {
  if (!ragToggleBtn || !ragToggleLabel) return;
  if (useRag) {
    ragToggleBtn.classList.remove('off');
    ragToggleBtn.classList.add('on');
    ragToggleBtn.title = '使用知识库检索（当前：开）。点击关闭';
    ragToggleLabel.textContent = '知识库';
    if (statusBadge) {
      statusBadge.classList.remove('off');
    }
    if (statusLabel && !sendBtn.classList.contains('loading')) {
      statusLabel.textContent = 'RAG 服务已连接';
    }
  } else {
    ragToggleBtn.classList.remove('on');
    ragToggleBtn.classList.add('off');
    ragToggleBtn.title = '不使用知识库（当前：关）。点击开启';
    ragToggleLabel.textContent = '知识库';
    // 关闭知识库时仅改变右上角徽标样式与文案
    if (statusBadge) {
      statusBadge.classList.add('off');
    }
    if (statusLabel && !sendBtn.classList.contains('loading')) {
      statusLabel.textContent = '知识库未使用';
    }
  }
}

if (ragToggleBtn) {
  ragToggleBtn.addEventListener('click', function () {
    useRag = !useRag;
    updateRagToggleUI();
  });
  updateRagToggleUI();
}




sendBtn.addEventListener('click', sendQuestion);

inputEl.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

document.querySelectorAll('.suggestion[data-question]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const q = btn.getAttribute('data-question') || '';
    if (!q) return;
    inputEl.value = q;
    inputEl.focus();
  });
});

/* 初始化：加载 sessions；若为空则创建一个 */
(async function initSessions() {
  try {
    await loadSessions();
    if (!activeSessionId) {
      const s = await createNewSession();
      await switchToSession(s.id);
      return;
    }
    await switchToSession(activeSessionId);
  } catch (e) {
    console.error(e);
  }
})();

/* 新聊天按钮：若当前会话未输入任何内容，则不新建 */
if (navNewChatBtn) {
  navNewChatBtn.addEventListener('click', async function () {
    const sid = activeSessionId;
    const msgs = sid ? messagesCache.get(sid) || [] : [];
    const hasUserInput = msgs.some((m) => m.role === 'user');
    if (!hasUserInput) return;
    const newSess = await createNewSession();
    await switchToSession(newSess.id);
  });
}

/* 点击 session 项切换会话 */
if (navSessionsList) {
  navSessionsList.addEventListener('click', function (e) {
    const target = e.target;

    // 点击菜单项：重命名/删除
    const menuActionBtn = target.closest('.nav-session-menu-item');
    if (menuActionBtn) {
      const item = target.closest('.nav-session-item');
      if (!item) return;
      const id = item.dataset.sessionId;
      if (!id) return;
      const action = menuActionBtn.dataset.action;
      const sess = sessions.find((s) => s.id === id);
      if (!sess) return;

      if (action === 'rename') {
        const current = sess.name || '';
        const next = window.prompt('重命名会话：', current);
        if (typeof next === 'string') {
          const trimmed = next.trim();
          if (trimmed) {
            const title = trimmed.length > 30 ? trimmed.slice(0, 30) + '…' : trimmed;
            apiJson('/api/sessions/' + encodeURIComponent(id), {
              method: 'PATCH',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ title: title })
            })
              .then(loadSessions)
              .catch((err) => alert(err.message || '重命名失败'));
          }
        }
        closeAllSessionMenus();
        return;
      }

      if (action === 'delete') {
        const ok = window.confirm('确定删除该会话吗？');
        if (!ok) return;
        apiJson('/api/sessions/' + encodeURIComponent(id), { method: 'DELETE' })
          .then(async function () {
            await loadSessions();
            // 删除当前会话时：切换到新的 active；若为空则创建一个
            if (!activeSessionId) {
              const snew = await createNewSession();
              await switchToSession(snew.id);
              return;
            }
            await switchToSession(activeSessionId);
          })
          .catch((err) => alert(err.message || '删除失败'));
        closeAllSessionMenus();
        return;
      }

      return;
    }

    // 点击菜单按钮本身，不切换会话（菜单按钮 handler 会处理开关）
    if (target.closest('.nav-session-menu-btn')) return;
    if (target.closest('.nav-session-menu')) return;

    // 点击会话主体：切换
    const item = target.closest('.nav-session-item');
    if (!item) return;
    const id = item.dataset.sessionId;
    if (id) switchToSession(id).catch((err) => console.error(err));
  });
}

/* 侧边栏折叠/展开 */
const navSidebar = document.querySelector('.nav-sidebar');
const sidebarToggleBtn = document.querySelector('.nav-sidebar-toggle');
const sidebarOpenTab = document.querySelector('.nav-sidebar-open-tab');
if (navSidebar && sidebarToggleBtn) {
  sidebarToggleBtn.addEventListener('click', function () {
    navSidebar.classList.add('collapsed');
    if (sidebarOpenTab) sidebarOpenTab.classList.add('visible');
  });
}
if (navSidebar && sidebarOpenTab) {
  sidebarOpenTab.addEventListener('click', function () {
    navSidebar.classList.remove('collapsed');
    sidebarOpenTab.classList.remove('visible');
  });
}

// 点击页面其它区域或按 Esc 时关闭菜单
document.addEventListener('click', function () {
  closeAllSessionMenus();
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') closeAllSessionMenus();
});


const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const sendLabel = document.getElementById('send-label');
const statusLabel = document.getElementById('status-label');
const statusBadge = document.querySelector('.badge');
const hintText = document.getElementById('hint-text');
const insightsEl = document.getElementById('insights');
const ragToggleBtn = document.getElementById('rag-toggle-btn');
const ragToggleLabel = document.getElementById('rag-toggle-label');

/** 是否使用知识库（RAG）检索，默认开启 */
let useRag = true;

const messages = [];

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

  if (role === 'user') {
    row.appendChild(body);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(body);
  }

  return { row, bubble };
}

function appendMessage(role, content) {
  messages.push({ role, content });

  const { row } = createMessageElements(role, content);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderInsights(data) {
  if (!insightsEl) return;

  const docs = Array.isArray(data.source_documents) ? data.source_documents : [];

  if (docs.length === 0) {
    insightsEl.innerHTML = '';
    return;
  }

  let html = '';

  if (docs.length > 0) {
    html += '<div class="insights-section">';
    html += '<span class="insights-title">引用文档</span>';
    html += '<div class="insights-doc-list">';

    docs.slice(0, 4).forEach((d, index) => {
      const name = d.doc_name || d.doc_path_name || '未命名文档';
      html +=
        '<button class="doc-chip" type="button">' +
        '<span class="doc-chip-index">' +
        (index + 1) +
        '</span>' +
        '<span class="doc-chip-name">' +
        name +
        '</span>' +
        '</button>';
    });

    html += '</div></div>';
  }

  insightsEl.innerHTML = html;
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

  appendMessage('user', text);
  inputEl.value = '';
  setLoading(true);

  const { row: botRow, bubble: botBubble } = createMessageElements('bot', '');
  botBubble.classList.add('streaming');
  botBubble.textContent = '';
  messagesEl.appendChild(botRow);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  let accumulated = '';
  renderInsights({ source_documents: [], token_usage: null });

  try {
    const resp = await fetch('/api/ask_stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, use_rerank: true, use_rag: useRag })
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
          renderInsights(msg.data);
        } else if (msg.type === 'error' && msg.data) {
          const errMsg = msg.data.error_message || '查询失败，请查看后端日志。';
          botBubble.textContent = '错误：' + errMsg;
          statusLabel.textContent = '后端返回错误';
          renderInsights({ source_documents: [], token_usage: null });
          return;
        }
      }
    }

    if (!accumulated) {
      botBubble.classList.remove('streaming');
      renderBotMarkdown(botBubble, '（后端未返回内容）', { typesetMath: false });
    }
  } catch (err) {
    console.error(err);
    botBubble.classList.remove('streaming');
    renderBotMarkdown(botBubble, '请求异常，请检查服务是否正常运行。', { typesetMath: false });
    statusLabel.textContent = '请求异常';
    renderInsights({ source_documents: [], token_usage: null });
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

appendMessage('bot', '你好，我是 Mildoc Chat，可以基于 Milvus + 文档知识库回答你的问题，并在下方展示引用的文档来源。');


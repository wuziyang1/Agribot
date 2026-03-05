const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const sendLabel = document.getElementById('send-label');
const statusLabel = document.getElementById('status-label');
const hintText = document.getElementById('hint-text');
const insightsEl = document.getElementById('insights');

const messages = [];

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
  bubble.textContent = content;

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
  const tokenUsage = data.token_usage || null;

  if (docs.length === 0 && !tokenUsage) {
    insightsEl.innerHTML = '';
    return;
  }

  let html = '<div class="insights-header"><span>本次回答基于以下文档与上下文</span></div>';

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

  if (tokenUsage) {
    html += '<div class="insights-section insights-tokens">';
    html += '<span class="insights-title">Token</span>';
    html += '<span class="insights-token-item">提示：' + (tokenUsage.prompt_tokens ?? '-') + '</span>';
    html += '<span class="insights-token-item">回答：' + (tokenUsage.completion_tokens ?? '-') + '</span>';
    html += '<span class="insights-token-item">总计：' + (tokenUsage.total_tokens ?? '-') + '</span>';
    html += '</div>';
  }

  insightsEl.innerHTML = html;
}

function setLoading(loading) {
  if (loading) {
    sendBtn.disabled = true;
    sendLabel.textContent = '思考中…';
    statusLabel.textContent = '正在生成回答…';
    if (hintText) {
      hintText.textContent = '正在处理你的问题，请稍候…';
    }
  } else {
    sendBtn.disabled = false;
    sendLabel.textContent = '发送';
    statusLabel.textContent = 'RAG 服务已连接';
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

  // 预先创建一个空的 AI 气泡，用于流式追加内容
  const { row: botRow, bubble: botBubble } = createMessageElements('bot', '');
  messagesEl.appendChild(botRow);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  let accumulated = '';
  renderInsights({ source_documents: [], token_usage: null });

  try {
    const resp = await fetch('/api/ask_stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, use_rerank: true })
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

    // 如果后端没返回任何 chunk，就给一个兜底文案
    if (!accumulated) {
      botBubble.textContent = '（后端未返回内容）';
    }
  } catch (err) {
    console.error(err);
    botBubble.textContent = '请求异常，请检查服务是否正常运行。';
    statusLabel.textContent = '请求异常';
    renderInsights({ source_documents: [], token_usage: null });
  } finally {
    setLoading(false);
  }
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

// 初始欢迎消息
appendMessage('bot', '你好，我是 Mildoc Chat，可以基于 Milvus + 文档知识库回答你的问题，并在下方展示引用的文档来源。');


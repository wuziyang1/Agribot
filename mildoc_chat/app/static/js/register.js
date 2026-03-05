document.addEventListener('DOMContentLoaded', function () {
  var emailInput = document.getElementById('email');
  var codeInput = document.getElementById('email_code');
  var usernameInput = document.getElementById('username');
  var passwordInput = document.getElementById('password');
  var sendBtn = document.getElementById('send-code-btn');
  var form = document.getElementById('register-form');

  if (emailInput) {
    emailInput.focus();
  }

  function postJson(url, data) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data || {})
    }).then(function (r) {
      return r.json().catch(function () {
        return { success: false, error_message: '响应解析失败' };
      });
    });
  }

  function setCountdown(seconds) {
    var remaining = seconds;
    sendBtn.disabled = true;
    sendBtn.textContent = remaining + 's';
    var timer = setInterval(function () {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(timer);
        sendBtn.disabled = false;
        sendBtn.textContent = '发送验证码';
        return;
      }
      sendBtn.textContent = remaining + 's';
    }, 1000);
  }

  if (sendBtn) {
    sendBtn.addEventListener('click', function () {
      var email = (emailInput && emailInput.value || '').trim();
      if (!email) {
        alert('请先输入邮箱');
        if (emailInput) emailInput.focus();
        return;
      }
      sendBtn.disabled = true;
      sendBtn.textContent = '发送中...';

      postJson('/api/register/send_code', { email: email }).then(function (resp) {
        if (resp && resp.success) {
          setCountdown(resp.cooldown_seconds || 60);
          if (codeInput) codeInput.focus();
        } else {
          sendBtn.disabled = false;
          sendBtn.textContent = '发送验证码';
          alert((resp && resp.error_message) || '发送失败');
        }
      }).catch(function () {
        sendBtn.disabled = false;
        sendBtn.textContent = '发送验证码';
        alert('发送失败，请稍后重试');
      });
    });
  }

  function trySubmitOnEnter(inputEl) {
    if (!inputEl || !form) return;
    inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        form.submit();
      }
    });
  }

  trySubmitOnEnter(codeInput);
  trySubmitOnEnter(usernameInput);
  trySubmitOnEnter(passwordInput);
});


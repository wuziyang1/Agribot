document.addEventListener('DOMContentLoaded', function () {
  var usernameInput = document.getElementById('fp-username');
  var emailInput = document.getElementById('fp-email');
  var codeInput = document.getElementById('fp-code');
  var newPwdInput = document.getElementById('fp-new-password');
  var submitBtn = document.getElementById('fp-submit');
  var sendBtn = document.getElementById('fp-send-code');
  var errorEl = document.getElementById('fp-error');

  if (usernameInput) {
    usernameInput.focus();
  }

  function showError(msg) {
    if (!errorEl) return;
    errorEl.textContent = msg || '';
    errorEl.style.display = msg ? 'block' : 'none';
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

  function sendCode() {
    var username = (usernameInput && usernameInput.value || '').trim();
    var email = (emailInput && emailInput.value || '').trim();
    if (!username) {
      showError('请先填写用户名');
      if (usernameInput) usernameInput.focus();
      return;
    }
    if (!email) {
      showError('请先填写邮箱');
      if (emailInput) emailInput.focus();
      return;
    }
    showError('');

    postJson('/api/forgot/send_code', { username: username, email: email }).then(function (resp) {
      if (resp && resp.success) {
        alert('验证码已发送到邮箱（或写入服务日志），请查收');
        if (codeInput) codeInput.focus();
      } else {
        showError((resp && resp.error_message) || '发送验证码失败');
      }
    }).catch(function () {
      showError('发送验证码失败，请稍后重试');
    });
  }

  function resetPassword() {
    var username = (usernameInput && usernameInput.value || '').trim();
    var email = (emailInput && emailInput.value || '').trim();
    var code = (codeInput && codeInput.value || '').trim();
    var newPwd = (newPwdInput && newPwdInput.value || '').trim();

    if (!username || !email || !code || !newPwd) {
      showError('请完整填写所有字段');
      return;
    }
    showError('');

    submitBtn.disabled = true;
    postJson('/api/forgot/reset', {
      username: username,
      email: email,
      verification_code: code,
      new_password: newPwd
    }).then(function (resp) {
      submitBtn.disabled = false;
      if (resp && resp.success) {
        alert('密码重置成功，请使用新密码登录');
        window.location.href = '/login';
      } else {
        showError((resp && resp.error_message) || '重置密码失败');
      }
    }).catch(function () {
      submitBtn.disabled = false;
      showError('重置密码失败，请稍后重试');
    });
  }

  if (submitBtn) {
    submitBtn.addEventListener('click', resetPassword);
  }

  if (sendBtn) {
    sendBtn.addEventListener('click', sendCode);
  }

  // 支持在验证码输入框中按回车直接发送验证码（如果还没获取）
  if (codeInput) {
    codeInput.addEventListener('focus', function () {
      // 当用户聚焦验证码时，如果还没申请过验证码，引导发送
      if (!codeInput.value) {
        // 不自动发送，避免惊扰；仅清理错误
        showError('');
      }
    });
  }

  // 在新密码输入框按 Enter 直接提交
  if (newPwdInput) {
    newPwdInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        resetPassword();
      }
    });
  }

  // 在邮箱输入框失焦后自动提示发送验证码
  if (emailInput) {
    emailInput.addEventListener('blur', function () {
      // 不自动发送验证码，只是清错误信息
      showError('');
    });
  }

  // 在用户名或邮箱输入框上双击快速发送验证码（开发调试方便）
  [usernameInput, emailInput].forEach(function (el) {
    if (!el) return;
    el.addEventListener('dblclick', function () {
      sendCode();
    });
  });
});


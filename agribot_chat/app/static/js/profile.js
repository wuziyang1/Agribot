(function () {
  const editables = document.querySelectorAll('.profile-editable');
  editables.forEach(function (dd) {
    const field = dd.dataset.field;
    const textEl = dd.querySelector('.profile-editable-text');
    const inputEl = dd.querySelector('.profile-editable-input');

    function startEdit() {
      dd.classList.add('editing');
      inputEl.value = textEl.textContent === '未设置' ? '' : textEl.textContent;
      inputEl.focus();
    }

    function endEdit(save) {
      dd.classList.remove('editing');
      if (!save) return;
      const val = inputEl.value.trim();
      if (field === 'username' && !val) return;
      const payload = {};
      payload[field] = val;

      fetch('/api/profile/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
        .then(function (r) {
          return r.json().then(function (data) {
            if (!r.ok) throw new Error(data.error_message || '更新失败');
            return data;
          });
        })
        .then(function () {
          if (field === 'username') {
            textEl.textContent = val || '未设置';
            const titleEl = document.querySelector('.profile-title');
            if (titleEl) titleEl.textContent = val || '未设置';
            const avatarEl = document.querySelector('.profile-avatar');
            if (avatarEl) avatarEl.textContent = (val || 'M')[0].toUpperCase();
          } else {
            textEl.textContent = val || '未设置';
          }
          inputEl.value = val || '';
        })
        .catch(function (err) {
          alert(err.message || '更新失败');
        });
    }

    dd.addEventListener('click', function (e) {
      if (dd.classList.contains('editing')) return;
      if (e.target === inputEl) return;
      startEdit();
    });

    inputEl.addEventListener('blur', function () {
      if (!dd.classList.contains('editing')) return;
      endEdit(true);
    });

    inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        inputEl.blur();
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        inputEl.value = textEl.textContent === '未设置' ? '' : textEl.textContent;
        endEdit(false);
      }
    });
  });
})();

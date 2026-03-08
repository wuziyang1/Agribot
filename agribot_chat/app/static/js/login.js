document.addEventListener('DOMContentLoaded', function () {
  var usernameInput = document.getElementById('username');
  var passwordInput = document.getElementById('password');
  var form = document.getElementById('login-form');

  if (usernameInput) {
    usernameInput.focus();
  }

  if (passwordInput && form) {
    passwordInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        form.submit();
      }
    });
  }
});


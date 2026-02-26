// ========== API Client & Authentication ==========

const API_BASE = '/api';
let authRequired = false;

async function checkAuthRequired() {
  try {
    const resp = await fetch(API_BASE + '/check-auth', { method: 'GET' });
    if (resp.ok) {
      const data = await resp.json();
      if (data.status === 'ok') {
        authRequired = data.data?.auth_required || false;
      }
    }
  } catch(e) {}
}

function getToken() {
  try { return localStorage.getItem('iris_token') || ''; } catch(e) { return ''; }
}

function setToken(token) {
  try { localStorage.setItem('iris_token', token); } catch(e) {}
}

function clearToken() {
  try { localStorage.removeItem('iris_token'); } catch(e) {}
}

function showLoginModal() {
  document.getElementById('login-modal').classList.add('show');
  document.getElementById('access-key-input').focus();
}

function hideLoginModal() {
  document.getElementById('login-modal').classList.remove('show');
}

async function doLogin() {
  const key = document.getElementById('access-key-input').value;
  if (!key) {
    toast('请输入访问密钥', 'error');
    return;
  }

  try {
    const resp = await fetch(API_BASE + '/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key })
    });
    const data = await resp.json();
    if (data.status === 'ok' && data.data?.token) {
      setToken(data.data.token);
      hideLoginModal();
      toast('登录成功', 'success');
      loadDashboard();
    } else {
      toast(data.message || '登录失败', 'error');
    }
  } catch(e) {
    toast(`登录失败: ${e.message}`, 'error');
  }
}

let lastFailedRequest = null;

async function api(path, options = {}) {
  const url = API_BASE + path;
  const headers = { 'Content-Type': 'application/json' };
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;

  try {
    const resp = await fetch(url, { ...options, headers: { ...headers, ...options.headers } });
    if (resp.status === 401) {
      if (authRequired) {
        showLoginModal();
      } else {
        toast('认证失败，请重新登录', 'error');
      }
      return null;
    }
    const ct = resp.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      return await resp.json();
    }
    return await resp.text();
  } catch(e) {
    lastFailedRequest = { path, options };
    toast(`请求失败: ${e.message}`, 'error', true);
    return null;
  }
}

// ========== Toast ==========
function toast(msg, type = 'info', retryable = false) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = escHtml(msg);
  if (retryable && lastFailedRequest) {
    const retryBtn = document.createElement('button');
    retryBtn.className = 'toast-retry';
    retryBtn.textContent = '重试';
    retryBtn.onclick = () => {
      el.remove();
      if (lastFailedRequest) {
        api(lastFailedRequest.path, lastFailedRequest.options);
        lastFailedRequest = null;
      }
    };
    el.appendChild(retryBtn);
  }
  container.appendChild(el);
  setTimeout(() => el.remove(), retryable ? 8000 : 4000);
}

// ========== Modal ==========
let modalCallback = null;

function showConfirm(title, message, callback) {
  document.getElementById('confirm-title').textContent = title;
  document.getElementById('confirm-message').textContent = message;
  document.getElementById('confirm-modal').classList.add('show');
  modalCallback = callback;
  document.getElementById('confirm-btn').onclick = async () => {
    const cb = modalCallback;
    closeModal('confirm-modal');
    if(cb) {
      try {
        await cb();
      } catch (e) {
        console.error('Confirm callback error:', e);
        toast('操作失败: ' + (e.message || '未知错误'), 'error');
      }
    }
  };
}

function closeModal(id) {
  document.getElementById(id).classList.remove('show');
  if (id === 'confirm-modal') modalCallback = null;
}

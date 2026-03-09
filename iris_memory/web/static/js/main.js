/**
 * Iris Memory Dashboard — 主入口
 * ES6 模块化，统一导航与认证管理
 */
import { api } from './api/client.js';
import { store } from './store/index.js';
import { toast } from './components/toast.js';
import { closeModal } from './components/modal.js';

import { loadDashboard, loadTrend } from './pages/dashboard.js';
import {
  searchMemories, memPage, toggleSelectAll, toggleSelect, batchDelete,
  resetFilters as resetMemoryFilters, changePageSize as memChangePageSize,
  getState as memGetState,
} from './pages/memories.js';
import {
  initKg, switchKgTab, loadKgGraph, searchKgNodes, searchKgEdges,
  refreshKgTab, hideNodePopup,
} from './pages/kg.js';
import {
  searchPersonas, loadPersonas, resetPersonaFilters,
  changePageSize as personaChangePageSize, getState as personaGetState,
} from './pages/personas.js';
import {
  loadProactiveStatus, addWhitelist, checkWhitelist, refreshProactiveTab,
  getState as proactiveGetState,
} from './pages/proactive.js';
import { switchIoTab, exportMemories, exportKg, handleFileDrop, handleFileSelect } from './pages/io.js';
import { loadCooldown } from './pages/cooldown.js';
import { loadLlm } from './pages/llm.js';
import { loadConfig, filterConfig, showDiff, exportSnapshot } from './pages/config.js';
import { loadSystem } from './pages/system.js';

const pageLoaders = {
  dashboard: loadDashboard,
  memories: () => { if (!memGetState().loaded) searchMemories(); },
  kg: () => { initKg(); },
  personas: () => { if (!personaGetState().loaded) searchPersonas(); },
  proactive: () => { if (!proactiveGetState().loaded) loadProactiveStatus(); },
  io: () => {},
  cooldown: loadCooldown,
  llm: loadLlm,
  config: loadConfig,
  system: loadSystem,
};

function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  const sec = document.getElementById(`sec-${name}`);
  if (sec) sec.classList.add('active');
  document.querySelector(`.nav-item[data-section="${name}"]`)?.classList.add('active');

  store.set('currentPage', name);
  const loader = pageLoaders[name];
  if (loader) loader();
}

function showLoginModal() {
  document.getElementById('login-modal').classList.add('show');
  document.getElementById('access-key-input')?.focus();
}

function hideLoginModal() {
  document.getElementById('login-modal').classList.remove('show');
}

async function doLogin() {
  const input = document.getElementById('access-key-input');
  const key = input?.value?.trim();
  if (!key) { toast.err('请输入访问密钥'); return; }

  const { ok, msg } = await api.login(key);
  if (ok) {
    hideLoginModal();
    toast.ok(msg);
    loadDashboard();
  } else {
    toast.err(msg);
  }
}

async function init() {
  const { authRequired, authenticated } = await api.checkAuth();
  if (authRequired && !api.hasToken) {
    showLoginModal();
  } else {
    showSection('dashboard');
  }
}

function bindEvents() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      showSection(item.dataset.section);
    });
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-overlay.show').forEach(m => m.classList.remove('show'));
      hideNodePopup();
    }
  });

  window.addEventListener('auth:required', showLoginModal);
  window.addEventListener('api:error', e => {
    toast.err(`请求失败: ${e.detail?.error?.message || '网络错误'}`);
  });
}

Object.assign(window, {
  showSection,
  doLogin,
  loadDashboard, loadTrend,
  searchMemories, memPage, toggleSelectAll, toggleSelect,
  batchDeleteMemories: batchDelete,
  resetMemoryFilters, exportSelectedMemories: () => toast.info('请使用导出功能'),
  switchKgTab, loadKgGraph, searchKgNodes, searchKgEdges, refreshKgTab,
  hideNodePopup,
  searchPersonas, loadPersonas, resetPersonaFilters,
  loadProactiveStatus, addProactiveWhitelist: addWhitelist,
  checkProactiveWhitelist: checkWhitelist, refreshProactiveTab,
  switchIoTab, exportMemories, exportKg, handleFileDrop, handleFileSelect,
  loadCooldown,
  loadLlm,
  loadConfig, filterConfig, showConfigDiff: showDiff, exportConfigSnapshot: exportSnapshot,
  loadSystem,
  closeModal,
});

function bindPageSizeSelects() {
  const memPs = document.getElementById('mem-page-size');
  if (memPs) memPs.addEventListener('change', () => memChangePageSize(memPs.value));

  const personaPs = document.getElementById('persona-page-size');
  if (personaPs) personaPs.addEventListener('change', () => personaChangePageSize(personaPs.value));
}

document.addEventListener('DOMContentLoaded', () => {
  bindEvents();
  bindPageSizeSelects();
  init();
});

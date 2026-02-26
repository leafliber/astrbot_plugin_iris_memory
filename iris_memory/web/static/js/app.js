// ========== Application Entry Point ==========

// Navigation
function showSection(name, navEl) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('sec-' + name).classList.add('active');
  if (navEl) navEl.classList.add('active');
  else document.querySelector(`.nav-item[data-section="${name}"]`)?.classList.add('active');

  if (name === 'dashboard') loadDashboard();
  if (name === 'memories') { if (!memState.loaded) searchMemories(); }
  if (name === 'personas') loadPersonas();
}

// Initialization
async function init() {
  await checkAuthRequired();
  if (authRequired && !getToken()) {
    showLoginModal();
  } else {
    loadDashboard();
  }
}

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
  // Escape closes modals
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.show').forEach(m => m.classList.remove('show'));
    hideNodePopup();
  }
});

// Boot
document.addEventListener('DOMContentLoaded', () => {
  init();
});

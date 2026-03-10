/**
 * 确认弹窗 + 通用 Modal 管理
 */
import { esc } from '../utils/escape.js';

/**
 * 显示确认弹窗
 * @param {string} title
 * @param {string} message
 * @param {Function} onConfirm
 * @param {'danger'|'primary'} btnClass
 */
export function showConfirm(title, message, onConfirm, btnClass = 'danger') {
  const overlay = document.getElementById('confirm-modal');
  overlay.querySelector('#confirm-title').textContent = title;
  overlay.querySelector('#confirm-message').textContent = message;
  const btn = overlay.querySelector('#confirm-btn');
  btn.className = `btn btn-${btnClass}`;
  btn.onclick = () => { closeModal('confirm-modal'); onConfirm(); };
  overlay.classList.add('show');
}

/**
 * 关闭指定 modal
 * @param {string} id
 */
export function closeModal(id) {
  document.getElementById(id)?.classList.remove('show');
}

/**
 * 打开指定 modal
 * @param {string} id
 */
export function openModal(id) {
  document.getElementById(id)?.classList.add('show');
}

/**
 * 在 modal body 中设置内容并显示
 * @param {string} modalId
 * @param {string} html  已经转义后的 HTML
 */
export function showDetailModal(modalId, html) {
  const overlay = document.getElementById(modalId);
  const body = overlay.querySelector('.modal-body');
  if (body) body.innerHTML = html;
  overlay.classList.add('show');
}

/**
 * 初始化 modal 点击空白处关闭
 */
export function initModalClose() {
  document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        overlay.classList.remove('show');
      }
    });
  });
}

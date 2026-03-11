/**
 * 知识图谱页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { nodeColors, nodeTypeLabels, relationLabels, fmtTime } from '../utils/format.js';
import { toast } from '../components/toast.js';
import { showConfirm } from '../components/modal.js';
import { renderPagination } from '../components/pagination.js';

// ── 图谱可视化状态 ──
let graphNodes = [];
let graphEdges = [];
let canvas, ctx;
let offsetX = 0, offsetY = 0, scale = 1;
let dragging = null, panning = false, panStart = { x: 0, y: 0 };

// ── 分页状态 ──
const nodesState = { page: 1, pageSize: 20, total: 0 };
const edgesState = { page: 1, pageSize: 20, total: 0 };

// ── 加载状态 ──
const loadedState = { graph: false, nodes: false, edges: false };

export function initKg() {
  canvas = document.getElementById('kg-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');
  setupInteraction();
}

// ── Tab 切换 ──
export function switchKgTab(tab) {
  ['graph', 'nodes', 'edges'].forEach(t => {
    const el = document.getElementById(`kg-tab-${t}`);
    if (el) el.style.display = t === tab ? '' : 'none';
  });
  document.querySelectorAll('#sec-kg .tab').forEach((el, i) => {
    el.classList.toggle('active', ['graph', 'nodes', 'edges'][i] === tab);
  });

  if (tab === 'graph') {
    drawGraph();
    if (!loadedState.graph) {
      loadedState.graph = true;
      loadKgGraph();
    }
  } else if (tab === 'nodes' && !loadedState.nodes) {
    loadedState.nodes = true;
    searchKgNodes();
  } else if (tab === 'edges' && !loadedState.edges) {
    loadedState.edges = true;
    searchKgEdges();
  }
}

export function refreshKgTab() {
  const active = document.querySelector('#sec-kg .tab.active');
  const tab = active?.textContent.includes('节点') ? 'nodes' : active?.textContent.includes('边') ? 'edges' : 'graph';
  if (tab === 'graph') loadKgGraph();
  else if (tab === 'nodes') searchKgNodes();
  else searchKgEdges();
}

// ── 图谱加载与布局 ──
export async function loadKgGraph() {
  const uid = val('kg-user'), center = val('kg-center'), depth = val('kg-depth');
  showGraphLoading(true);

  const res = await api.get('/kg/graph', { user_id: uid, group_id: '', center_node_id: center, depth, max_nodes: 200 });
  showGraphLoading(false);

  if (!res || res.status !== 'ok') { toast.err('加载图谱失败'); return; }

  graphNodes = (res.data?.nodes || []).map(n => ({
    ...n, x: Math.random() * 600 + 100, y: Math.random() * 400 + 50,
    vx: 0, vy: 0, r: Math.max(8, Math.min(25, (n.size || 5) * 3)),
  }));
  graphEdges = res.data?.edges || [];

  el('kg-graph-info').textContent = `${graphNodes.length} 个节点, ${graphEdges.length} 条边`;

  if (graphNodes.length) layout();
  drawGraph();
}

function layout() {
  const iterations = Math.min(200, Math.max(50, graphNodes.length * 2));
  for (let iter = 0; iter < iterations; iter++) {
    const alpha = 1 - iter / iterations;
    // 斥力
    for (let i = 0; i < graphNodes.length; i++) {
      for (let j = i + 1; j < graphNodes.length; j++) {
        const a = graphNodes[i], b = graphNodes[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let force = 5000 / (dist * dist);
        const fx = dx / dist * force * alpha, fy = dy / dist * force * alpha;
        a.vx -= fx; a.vy -= fy; b.vx += fx; b.vy += fy;
      }
    }
    // 引力
    const nodeMap = new Map(graphNodes.map(n => [n.id, n]));
    for (const e of graphEdges) {
      const a = nodeMap.get(e.source), b = nodeMap.get(e.target);
      if (!a || !b) continue;
      let dx = b.x - a.x, dy = b.y - a.y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 1;
      let force = (dist - 100) * 0.05 * alpha;
      const fx = dx / dist * force, fy = dy / dist * force;
      a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
    }
    // 应用
    for (const n of graphNodes) {
      n.x += n.vx * 0.5; n.y += n.vy * 0.5;
      n.vx *= 0.8; n.vy *= 0.8;
    }
  }
  // 居中
  if (graphNodes.length) {
    const cx = graphNodes.reduce((s, n) => s + n.x, 0) / graphNodes.length;
    const cy = graphNodes.reduce((s, n) => s + n.y, 0) / graphNodes.length;
    const w = canvas.width / 2, h = canvas.height / 2;
    graphNodes.forEach(n => { n.x += w - cx; n.y += h - cy; });
  }
}

function drawGraph() {
  if (!canvas || !ctx) return;
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  const nodeMap = new Map(graphNodes.map(n => [n.id, n]));

  // 绘制边
  ctx.lineWidth = 1;
  for (const e of graphEdges) {
    const a = nodeMap.get(e.source), b = nodeMap.get(e.target);
    if (!a || !b) continue;
    ctx.strokeStyle = 'rgba(124,108,240,0.3)';
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    // 箭头
    const dx = b.x - a.x, dy = b.y - a.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const arrowX = b.x - dx / dist * b.r, arrowY = b.y - dy / dist * b.r;
    const angle = Math.atan2(dy, dx);
    ctx.fillStyle = 'rgba(124,108,240,0.5)';
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - 8 * Math.cos(angle - 0.3), arrowY - 8 * Math.sin(angle - 0.3));
    ctx.lineTo(arrowX - 8 * Math.cos(angle + 0.3), arrowY - 8 * Math.sin(angle + 0.3));
    ctx.closePath(); ctx.fill();
    // 边标签
    if (e.label) {
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      ctx.fillStyle = 'rgba(139,143,168,0.8)';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(relationLabels[e.label] || e.label, mx, my - 4);
    }
  }

  // 绘制节点
  for (const n of graphNodes) {
    const color = nodeColors[n.type] || nodeColors.unknown;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // 标签
    ctx.fillStyle = '#e4e6f0';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(n.label || '', n.x, n.y + n.r + 14);
  }

  ctx.restore();
}

// ── 交互 ──
function setupInteraction() {
  if (!canvas) return;

  canvas.addEventListener('mousedown', e => {
    const { mx, my } = mousePos(e);
    const hit = graphNodes.find(n => Math.hypot(n.x - mx, n.y - my) <= n.r);
    if (hit) { dragging = hit; canvas.style.cursor = 'grabbing'; }
    else { panning = true; panStart = { x: e.clientX - offsetX, y: e.clientY - offsetY }; canvas.style.cursor = 'move'; }
  });

  canvas.addEventListener('mousemove', e => {
    if (dragging) {
      const { mx, my } = mousePos(e);
      dragging.x = mx; dragging.y = my;
      drawGraph();
    } else if (panning) {
      offsetX = e.clientX - panStart.x; offsetY = e.clientY - panStart.y;
      drawGraph();
    }
  });

  canvas.addEventListener('mouseup', () => {
    dragging = null; panning = false; canvas.style.cursor = 'default';
  });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    scale = Math.max(0.1, Math.min(5, scale * factor));
    drawGraph();
  }, { passive: false });

  canvas.addEventListener('click', e => {
    if (panning) return;
    const { mx, my } = mousePos(e);
    const hit = graphNodes.find(n => Math.hypot(n.x - mx, n.y - my) <= n.r);
    if (hit) showNodePopup(hit, e);
    else hideNodePopup();
  });
}

function mousePos(e) {
  const rect = canvas.getBoundingClientRect();
  return { mx: (e.clientX - rect.left - offsetX) / scale, my: (e.clientY - rect.top - offsetY) / scale };
}

function showNodePopup(node, e) {
  const popup = document.getElementById('node-popup');
  popup.innerHTML = `
    <h4>${esc(node.label)}</h4>
    <div class="detail-item"><div class="detail-label">类型</div><div class="detail-value">${esc(nodeTypeLabels[node.type] || node.type)}</div></div>
    <div class="detail-item"><div class="detail-label">置信度</div><div class="detail-value">${node.confidence != null ? (node.confidence * 100).toFixed(0) + '%' : '-'}</div></div>
    <div style="margin-top:8px;display:flex;gap:6px">
      <button class="btn btn-outline btn-sm" onclick="window.__kg.focusNode('${esc(node.id)}')">聚焦</button>
      <button class="btn btn-outline btn-sm" onclick="window.__kg.showNodeEdges('${esc(node.id)}')">查看边</button>
      <button class="btn btn-danger btn-sm" onclick="window.__kg.deleteNode('${esc(node.id)}')">删除</button>
    </div>`;
  popup.style.display = 'block';
  const rect = canvas.getBoundingClientRect();
  popup.style.left = (e.clientX - rect.left + 10) + 'px';
  popup.style.top = (e.clientY - rect.top + 10) + 'px';
}

export function hideNodePopup() {
  const popup = document.getElementById('node-popup');
  if (popup) popup.style.display = 'none';
}

export function focusNode(id) {
  el('kg-center').value = id;
  loadKgGraph();
}

export function showNodeEdges(nodeId) {
  switchKgTab('edges');
  el('kg-edge-node').value = nodeId;
  searchKgEdges();
}

// ── 节点表格 ──
export async function searchKgNodes() {
  const q = val('kg-node-query'), uid = val('kg-node-user'), nt = val('kg-node-type');
  const res = await api.get('/kg/nodes', {
    query: q, user_id: uid, node_type: nt,
    page: nodesState.page, page_size: nodesState.pageSize,
  });
  if (!res || res.status !== 'ok') return;

  const d = res.data;
  nodesState.total = d.total || 0;
  el('kg-nodes-total-info').textContent = `共 ${nodesState.total} 个节点`;

  const nodes = d.items || [];
  const tbody = document.getElementById('kg-nodes-tbody');
  if (!nodes.length) { tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text2)">暂无节点</td></tr>'; el('kg-nodes-pagination').innerHTML = ''; return; }

  tbody.innerHTML = nodes.map(n => `<tr>
    <td>${esc(n.display_name || n.name)}</td>
    <td><span style="color:${nodeColors[n.node_type] || nodeColors.unknown}">${esc(nodeTypeLabels[n.node_type] || n.node_type)}</span></td>
    <td>${esc(n.user_id || '-')}</td>
    <td>${n.mention_count ?? '-'}</td>
    <td>${n.confidence != null ? (n.confidence * 100).toFixed(0) + '%' : '-'}</td>
    <td>${esc(fmtTime(n.created_time))}</td>
    <td>
      <button class="btn btn-outline btn-sm" onclick="window.__kg.focusNode('${esc(n.id)}')">聚焦</button>
      <button class="btn btn-danger btn-sm" onclick="window.__kg.deleteNode('${esc(n.id)}')">删除</button>
    </td>
  </tr>`).join('');

  renderPagination({
    page: nodesState.page, pageSize: nodesState.pageSize, total: nodesState.total,
    onChange: p => { nodesState.page = p; searchKgNodes(); },
    container: el('kg-nodes-pagination'),
  });
}

export function kgNodesPage(p) { nodesState.page = p; searchKgNodes(); }
export function changeKgNodesPageSize(v) { nodesState.pageSize = Number(v); nodesState.page = 1; searchKgNodes(); }

// ── 边表格 ──
export async function searchKgEdges() {
  const uid = val('kg-edge-user'), nid = val('kg-edge-node'), rt = val('kg-edge-relation');
  const res = await api.get('/kg/edges', {
    user_id: uid, node_id: nid, relation_type: rt,
    page: edgesState.page, page_size: edgesState.pageSize,
  });
  if (!res || res.status !== 'ok') return;

  const d = res.data;
  edgesState.total = d.total || 0;
  el('kg-edges-total-info').textContent = `共 ${edgesState.total} 条边`;

  const edges = d.items || [];
  const tbody = document.getElementById('kg-edges-tbody');
  if (!edges.length) { tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text2)">暂无边</td></tr>'; el('kg-edges-pagination').innerHTML = ''; return; }

  tbody.innerHTML = edges.map(e => `<tr>
    <td>${esc(e.source_name || e.source_id)}</td>
    <td>${esc(relationLabels[e.relation_type] || e.relation_type)}</td>
    <td>${esc(e.target_name || e.target_id)}</td>
    <td>${e.confidence != null ? (e.confidence * 100).toFixed(0) + '%' : '-'}</td>
    <td>${e.weight ?? '-'}</td>
    <td>${esc(e.user_id || '-')}</td>
    <td>${esc(fmtTime(e.created_time))}</td>
    <td><button class="btn btn-danger btn-sm" onclick="window.__kg.deleteEdge('${esc(e.id)}')">删除</button></td>
  </tr>`).join('');

  renderPagination({
    page: edgesState.page, pageSize: edgesState.pageSize, total: edgesState.total,
    onChange: p => { edgesState.page = p; searchKgEdges(); },
    container: el('kg-edges-pagination'),
  });
}

export function kgEdgesPage(p) { edgesState.page = p; searchKgEdges(); }
export function changeKgEdgesPageSize(v) { edgesState.pageSize = Number(v); edgesState.page = 1; searchKgEdges(); }

export function deleteNode(id) {
  showConfirm('删除节点', '删除节点将同时删除关联的边，是否继续？', async () => {
    const res = await api.del(`/kg/nodes/${encodeURIComponent(id)}`);
    if (res?.status === 'ok') { toast.ok('已删除'); hideNodePopup(); loadKgGraph(); searchKgNodes(); }
    else toast.err(res?.message || '删除失败');
  });
}

export function deleteEdge(id) {
  showConfirm('删除边', '确定要删除此关系边吗？', async () => {
    const res = await api.del(`/kg/edges/${encodeURIComponent(id)}`);
    if (res?.status === 'ok') { toast.ok('已删除'); searchKgEdges(); }
    else toast.err(res?.message || '删除失败');
  });
}

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
function val(id) { return (el(id)?.value ?? '').trim(); }
function showGraphLoading(v) { const o = el('kg-graph-loading'); if (o) o.style.display = v ? 'flex' : 'none'; }

// window bridge
window.__kg = {
  focusNode, showNodeEdges, deleteNode, deleteEdge, hideNodePopup,
};

// ========== Knowledge Graph Section ==========

let currentKgTab = 'graph';

function switchKgTab(tab) {
  currentKgTab = tab;
  document.getElementById('kg-tab-graph').style.display = tab === 'graph' ? 'block' : 'none';
  document.getElementById('kg-tab-nodes').style.display = tab === 'nodes' ? 'block' : 'none';
  document.getElementById('kg-tab-edges').style.display = tab === 'edges' ? 'block' : 'none';
  document.querySelectorAll('#sec-kg .tab').forEach((t, i) => t.classList.toggle('active',
    (i===0&&tab==='graph')||(i===1&&tab==='nodes')||(i===2&&tab==='edges')));
  if (tab === 'nodes') searchKgNodes();
  if (tab === 'edges') searchKgEdges();
}

function refreshKgTab() {
  if (currentKgTab === 'graph') loadKgGraph();
  else if (currentKgTab === 'nodes') searchKgNodes();
  else if (currentKgTab === 'edges') searchKgEdges();
}

// ========== Graph Rendering (Canvas) ==========

let graphData = { nodes:[], edges:[] };
let graphNodes = [];
let dragNode = null;
let panOffset = { x: 0, y: 0 };
let isPanning = false;
let lastMouse = { x: 0, y: 0 };
let graphZoom = 1.0;

async function loadKgGraph() {
  const userId = document.getElementById('kg-user').value;
  const center = document.getElementById('kg-center').value;
  const depth = document.getElementById('kg-depth').value;

  let path = `/kg/graph?depth=${depth}&max_nodes=100`;
  if (userId) path += `&user_id=${encodeURIComponent(userId)}`;
  if (center) path += `&center=${encodeURIComponent(center)}`;

  document.getElementById('kg-graph-loading').style.display = 'flex';
  const resp = await api(path);
  document.getElementById('kg-graph-loading').style.display = 'none';

  if (!resp || resp.status !== 'ok') { toast('加载图谱失败', 'error'); return; }

  graphData = resp.data || { nodes:[], edges:[] };
  document.getElementById('kg-graph-info').textContent = `节点: ${graphData.nodes.length}, 边: ${graphData.edges.length}`;

  graphZoom = 1.0;
  initGraph();
}

function initGraph() {
  const canvas = document.getElementById('kg-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth;
  const H = 500;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  panOffset = { x: W/2, y: H/2 };

  if (graphData.nodes.length === 0) {
    ctx.fillStyle = '#5c6082';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('暂无图谱数据，请输入用户 ID 后点击查询', W/2, H/2);
    return;
  }

  // Layout: force-directed
  graphNodes = graphData.nodes.map((n, i) => ({
    ...n,
    x: (Math.cos(i / graphData.nodes.length * Math.PI * 2)) * Math.min(W, H) * 0.3,
    y: (Math.sin(i / graphData.nodes.length * Math.PI * 2)) * Math.min(W, H) * 0.3,
    vx: 0, vy: 0,
    r: n.size || 14
  }));

  const nodeMap = {};
  graphNodes.forEach(n => nodeMap[n.id] = n);

  // Adaptive iteration count based on node count
  const iterations = Math.min(200, Math.max(50, graphNodes.length * 3));

  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < graphNodes.length; i++) {
      for (let j = i+1; j < graphNodes.length; j++) {
        let dx = graphNodes[j].x - graphNodes[i].x;
        let dy = graphNodes[j].y - graphNodes[i].y;
        let d = Math.sqrt(dx*dx + dy*dy) || 1;
        let f = 5000 / (d * d);
        graphNodes[i].vx -= dx/d * f;
        graphNodes[i].vy -= dy/d * f;
        graphNodes[j].vx += dx/d * f;
        graphNodes[j].vy += dy/d * f;
      }
    }
    for (const e of graphData.edges) {
      const s = nodeMap[e.source], t = nodeMap[e.target];
      if (!s || !t) continue;
      let dx = t.x - s.x, dy = t.y - s.y;
      let d = Math.sqrt(dx*dx + dy*dy) || 1;
      let f = (d - 80) * 0.05;
      s.vx += dx/d * f; s.vy += dy/d * f;
      t.vx -= dx/d * f; t.vy -= dy/d * f;
    }
    const damping = 0.5 + (iter / iterations) * 0.4;
    for (const n of graphNodes) {
      n.x += n.vx * 0.3;
      n.y += n.vy * 0.3;
      n.vx *= damping;
      n.vy *= damping;
    }
  }

  drawGraph(canvas, ctx, W, H);
  setupGraphInteraction(canvas, ctx, W, H);
}

function setupGraphInteraction(canvas, ctx, W, H) {
  // Cleanup old listeners
  canvas.onmousedown = null; canvas.onmousemove = null; canvas.onmouseup = null;
  canvas.onmouseleave = null; canvas.onwheel = null; canvas.onclick = null;

  canvas.onmousedown = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (ev.clientX - rect.left - panOffset.x) / graphZoom;
    const my = (ev.clientY - rect.top - panOffset.y) / graphZoom;

    dragNode = graphNodes.find(n => Math.hypot(n.x - mx, n.y - my) < (n.r + 4) / graphZoom);
    if (dragNode) {
      canvas.style.cursor = 'grabbing';
    } else {
      isPanning = true;
      lastMouse = { x: ev.clientX, y: ev.clientY };
      canvas.style.cursor = 'move';
    }
  };

  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    if (dragNode) {
      dragNode.x = (ev.clientX - rect.left - panOffset.x) / graphZoom;
      dragNode.y = (ev.clientY - rect.top - panOffset.y) / graphZoom;
      drawGraph(canvas, ctx, W, H);
    } else if (isPanning) {
      panOffset.x += ev.clientX - lastMouse.x;
      panOffset.y += ev.clientY - lastMouse.y;
      lastMouse = { x: ev.clientX, y: ev.clientY };
      drawGraph(canvas, ctx, W, H);
    } else {
      const mx = (ev.clientX - rect.left - panOffset.x) / graphZoom;
      const my = (ev.clientY - rect.top - panOffset.y) / graphZoom;
      const hover = graphNodes.find(n => Math.hypot(n.x - mx, n.y - my) < n.r + 4);
      canvas.style.cursor = hover ? 'pointer' : 'default';
    }
  };

  canvas.onmouseup = (ev) => {
    if (dragNode) {
      dragNode = null;
    }
    isPanning = false;
    canvas.style.cursor = 'default';
  };

  canvas.onmouseleave = () => { dragNode = null; isPanning = false; };

  // Zoom with scroll wheel
  canvas.onwheel = (ev) => {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = ev.clientX - rect.left;
    const mouseY = ev.clientY - rect.top;

    const zoomFactor = ev.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.2, Math.min(5, graphZoom * zoomFactor));

    // Adjust pan to zoom toward mouse position
    panOffset.x = mouseX - (mouseX - panOffset.x) * (newZoom / graphZoom);
    panOffset.y = mouseY - (mouseY - panOffset.y) * (newZoom / graphZoom);

    graphZoom = newZoom;
    drawGraph(canvas, ctx, W, H);
  };

  // Click to show node details
  canvas.onclick = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (ev.clientX - rect.left - panOffset.x) / graphZoom;
    const my = (ev.clientY - rect.top - panOffset.y) / graphZoom;

    const clicked = graphNodes.find(n => Math.hypot(n.x - mx, n.y - my) < n.r + 4);
    if (clicked) {
      showNodePopup(clicked, ev.clientX - rect.left, ev.clientY - rect.top);
    } else {
      hideNodePopup();
    }
  };

  // Click outside to hide popup
  const container = canvas.parentElement;
  const originalOnClick = container.onclick;
  container.onclick = (ev) => {
    if (originalOnClick) originalOnClick.call(container, ev);
    const popup = document.getElementById('node-popup');
    if (popup.style.display === 'block' && !popup.contains(ev.target) && ev.target !== canvas) {
      hideNodePopup();
    }
  };
}

function showNodePopup(node, x, y) {
  const popup = document.getElementById('node-popup');
  const connectedEdges = graphData.edges.filter(e => e.source === node.id || e.target === node.id);

  popup.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <h4 style="display:flex;align-items:center;gap:8px;margin:0;">
        <span style="width:12px;height:12px;border-radius:50%;background:${nodeColors[node.type]||nodeColors.unknown};display:inline-block;"></span>
        ${escHtml(node.label || node.id)}
      </h4>
      <button class="btn btn-sm" onclick="hideNodePopup()" style="padding:2px 6px;font-size:12px;">✕</button>
    </div>
    <div class="detail-item"><div class="detail-label">类型</div><div class="detail-value">${nodeTypeLabels[node.type] || node.type}</div></div>
    <div class="detail-item"><div class="detail-label">置信度</div><div class="detail-value">${(node.confidence || 0).toFixed(2)}</div></div>
    <div class="detail-item"><div class="detail-label">连接数</div><div class="detail-value">${connectedEdges.length} 条边</div></div>
    ${connectedEdges.length > 0 ? `<div class="detail-item"><div class="detail-label">关系</div><div class="detail-value" style="font-size:12px;">${
      connectedEdges.slice(0, 5).map(e => escHtml(e.label || e.relation_type || '')).join(', ')
    }${connectedEdges.length > 5 ? '...' : ''}</div></div>` : ''}
    <div style="margin-top:8px;display:flex;gap:6px;">
      <button class="btn btn-outline btn-sm" onclick="focusNode('${node.id}')">从此展开</button>
      <button class="btn btn-danger btn-sm" onclick="deleteKgNode('${node.id}','${escHtml(node.label || '')}')">删除</button>
    </div>
  `;

  // Position popup
  const container = popup.parentElement;
  const containerRect = container.getBoundingClientRect();
  let left = x + 10;
  let top = y - 10;
  if (left + 300 > containerRect.width) left = x - 310;
  if (top + 200 > containerRect.height) top = containerRect.height - 210;
  if (top < 0) top = 10;

  popup.style.left = left + 'px';
  popup.style.top = top + 'px';
  popup.style.display = 'block';
}

function hideNodePopup() {
  document.getElementById('node-popup').style.display = 'none';
}

function drawGraph(canvas, ctx, W, H) {
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(panOffset.x, panOffset.y);
  ctx.scale(graphZoom, graphZoom);

  const nodeMap = {};
  graphNodes.forEach(n => nodeMap[n.id] = n);

  // Draw edges
  ctx.lineWidth = 1 / graphZoom;
  for (const e of graphData.edges) {
    const s = nodeMap[e.source], t = nodeMap[e.target];
    if (!s || !t) continue;
    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(t.x, t.y);
    ctx.strokeStyle = 'rgba(140,145,180,0.35)';
    ctx.stroke();

    if (e.label && graphZoom > 0.5) {
      const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2;
      ctx.font = `${10/graphZoom}px sans-serif`;
      ctx.fillStyle = '#8b8fa8';
      ctx.textAlign = 'center';
      ctx.fillText(e.label, mx, my - 4);
    }

    // Arrow
    const angle = Math.atan2(t.y - s.y, t.x - s.x);
    const arrowX = t.x - Math.cos(angle) * (t.r + 4);
    const arrowY = t.y - Math.sin(angle) * (t.r + 4);
    const arrowSize = 8 / Math.max(graphZoom, 0.5);
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - arrowSize * Math.cos(angle - 0.4), arrowY - arrowSize * Math.sin(angle - 0.4));
    ctx.lineTo(arrowX - arrowSize * Math.cos(angle + 0.4), arrowY - arrowSize * Math.sin(angle + 0.4));
    ctx.closePath();
    ctx.fillStyle = 'rgba(140,145,180,0.45)';
    ctx.fill();
  }

  // Draw nodes
  for (const n of graphNodes) {
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = nodeColors[n.type] || nodeColors.unknown;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 2 / graphZoom;
    ctx.stroke();

    if (graphZoom > 0.4) {
      ctx.font = `${12/graphZoom}px sans-serif`;
      ctx.fillStyle = '#e4e6f0';
      ctx.textAlign = 'center';
      ctx.fillText(n.label || '', n.x, n.y + n.r + 14/graphZoom);
    }
  }

  ctx.restore();
}

// ========== KG Node / Edge Tables ==========

async function searchKgNodes() {
  const query = document.getElementById('kg-node-query').value;
  const userId = document.getElementById('kg-node-user').value;
  const nodeType = document.getElementById('kg-node-type').value;

  let path = `/kg/nodes?limit=50`;
  if (query) path += `&q=${encodeURIComponent(query)}`;
  if (userId) path += `&user_id=${encodeURIComponent(userId)}`;
  if (nodeType) path += `&type=${nodeType}`;

  const resp = await api(path);
  if (!resp || resp.status !== 'ok') return;

  const nodes = resp.data || [];

  if (nodes.length === 0) {
    document.getElementById('kg-nodes-tbody').innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text2)">暂无节点数据</td></tr>';
    return;
  }

  document.getElementById('kg-nodes-tbody').innerHTML = nodes.map(n => `<tr>
    <td title="ID: ${n.id}">${escHtml(n.display_name || n.name)}</td>
    <td><span class="badge" style="background:${nodeColors[n.node_type]||'#5c6082'}33;color:${nodeColors[n.node_type]||'#5c6082'}">${nodeTypeLabels[n.node_type]||n.node_type}</span></td>
    <td>${escHtml(n.user_id || '')}</td>
    <td>${n.mention_count}</td>
    <td>${(n.confidence || 0).toFixed(2)}</td>
    <td style="font-size:12px;">${n.created_time ? new Date(n.created_time).toLocaleString('zh-CN') : ''}</td>
    <td style="white-space:nowrap;">
      <button class="btn btn-outline btn-sm" onclick="focusNode('${n.id}')" style="margin-right:4px;">定位</button>
      <button class="btn btn-outline btn-sm" onclick="showNodeEdges('${n.id}')" style="margin-right:4px;">查边</button>
      <button class="btn btn-danger btn-sm" onclick="deleteKgNode('${n.id}','${escHtml(n.display_name||n.name)}')">删除</button>
    </td>
  </tr>`).join('');
}

async function searchKgEdges() {
  const userId = document.getElementById('kg-edge-user').value;
  const nodeId = document.getElementById('kg-edge-node').value;
  const relType = document.getElementById('kg-edge-relation').value;

  let path = `/kg/edges?limit=50`;
  if (userId) path += `&user_id=${encodeURIComponent(userId)}`;
  if (nodeId) path += `&node_id=${encodeURIComponent(nodeId)}`;
  if (relType) path += `&relation_type=${relType}`;

  const resp = await api(path);
  if (!resp || resp.status !== 'ok') return;

  const edges = resp.data || [];

  if (edges.length === 0) {
    document.getElementById('kg-edges-tbody').innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text2)">暂无边数据</td></tr>';
    return;
  }

  document.getElementById('kg-edges-tbody').innerHTML = edges.map(e => `<tr>
    <td title="ID: ${escHtml(e.source_id)}">${escHtml(e.source_name)}</td>
    <td><span class="badge" style="background:rgba(124,108,240,0.15);color:var(--accent);">${escHtml(relationLabels[e.relation_type] || e.relation_label || e.relation_type)}</span></td>
    <td title="ID: ${escHtml(e.target_id)}">${escHtml(e.target_name)}</td>
    <td>${(e.confidence || 0).toFixed(2)}</td>
    <td>${(e.weight || 0).toFixed(2)}</td>
    <td>${escHtml(e.user_id || '')}</td>
    <td style="font-size:12px;">${e.created_time ? new Date(e.created_time).toLocaleString('zh-CN') : ''}</td>
    <td>
      <button class="btn btn-danger btn-sm" onclick="deleteKgEdge('${e.id}','${escHtml(e.source_name)} → ${escHtml(e.target_name)}')">删除</button>
    </td>
  </tr>`).join('');
}

function showNodeEdges(nodeId) {
  document.getElementById('kg-edge-node').value = nodeId;
  switchKgTab('edges');
}

function focusNode(nodeId) {
  document.getElementById('kg-center').value = nodeId;
  switchKgTab('graph');
  loadKgGraph();
}

function deleteKgNode(id, name) {
  hideNodePopup();
  showConfirm('删除节点', `确定要删除节点 "${name}" 及其所有关联边吗？`, async () => {
    const resp = await api('/kg/node/delete', { method:'POST', body:JSON.stringify({id}) });
    if (resp && resp.status === 'ok') {
      toast(resp.message || '删除成功', 'success');
      searchKgNodes();
      if (currentKgTab === 'edges') searchKgEdges();
    } else {
      toast(resp?.message || '删除失败', 'error');
    }
  });
}

function deleteKgEdge(id, label) {
  showConfirm('删除边', `确定要删除关系 "${label}" 吗？`, async () => {
    const resp = await api('/kg/edge/delete', { method:'POST', body:JSON.stringify({id}) });
    if (resp && resp.status === 'ok') {
      toast('删除成功', 'success');
      searchKgEdges();
    } else {
      toast(resp?.message || '删除失败', 'error');
    }
  });
}

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["ready", "disabled", "error"])
def test_diagnostics_page_render_tabs_refresh_and_safe_text(tmp_path, mode):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for Pages interaction test")
    root = Path(__file__).parents[1]
    script = (root / "pages/status/diagnostics.js").read_text()
    html = (root / "pages/status/index.html").read_text()
    for name in (
        "diagnostics-refresh",
        "diagnostics-notice",
        "diagnostics-summary",
        "diagnostics-logs",
        "diagnostics-prepares",
        "diagnostics-sessions",
    ):
        assert f'id="{name}"' in html
    harness = r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
class Element {
  constructor() { this.children=[]; this.handlers={}; this.attrs={}; this.hidden=false; this.disabled=false; this.text=''; }
  set textContent(value) { this.text=String(value); this.children=[]; }
  get textContent() { return this.text + this.children.map(x=>x.textContent).join(''); }
  set innerHTML(value) { throw new Error('Unsafe HTML insertion'); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=nodes; this.text=''; }
  addEventListener(name,callback) { this.handlers[name]=callback; }
  setAttribute(name,value) { this.attrs[name]=String(value); }
  focus() { this.focused=true; }
}
const ids=new Map();
globalThis.document={getElementById(id){ if(!ids.has(id))ids.set(id,new Element()); return ids.get(id); },createElement(){return new Element();}};
let calls=0;
const mode=process.argv[3];
const data={enabled:mode!=='disabled',started_at:'2030-01-01T00:00:00Z',limits:{logs:200,prepares:50,sessions:200},
 logs:[{at:'2030-01-01T00:00:00Z',code:'<script>example</script>'}],
 prepares:[{at:'2030-01-01T00:00:00Z',entry_ref:'entry-test',purpose:'proactive',outcome:'ok',duration_ms:25,partitions:{memories:2},judgment:'degraded',entry_profile_version:3,entry_profile_stale:true}],
 sessions:[{session_ref:'session-test',entry_ref:'entry-test',since:'2030-01-01T00:00:00Z',counts:{message:{cooldown:2},calls:{chime_in:1},outcome:{sent:1}}}]};
globalThis.window={AstrBotPluginPage:{async ready(){},async apiGet(route){assert.equal(route,'diagnostics');calls++;if(mode==='error')throw new Error('iris_test_token');return data;}}};
await import('data:text/javascript;base64,'+Buffer.from(fs.readFileSync(process.argv[2],'utf8')).toString('base64'));
assert.equal(calls,1);
assert.equal(document.getElementById('diagnostics-refresh').disabled,false);
assert.ok(![...ids.values()].map(x=>x.textContent).join('').includes('iris_test_token'));
if(mode==='ready') {
 assert.ok(document.getElementById('diagnostics-logs').textContent.includes('<script>example</script>'));
 assert.ok(document.getElementById('diagnostics-prepares').textContent.includes('降级'));
 assert.ok(document.getElementById('diagnostics-prepares').textContent.includes('25'));
 assert.ok(document.getElementById('diagnostics-sessions').textContent.includes('cooldown'));
 document.getElementById('diagnostics-tab-prepares').handlers.click();
 assert.equal(document.getElementById('diagnostics-panel-logs').hidden,true);
 assert.equal(document.getElementById('diagnostics-panel-prepares').hidden,false);
 assert.equal(document.getElementById('diagnostics-tab-prepares').attrs['aria-selected'],'true');
} else if(mode==='disabled') {
 assert.ok(document.getElementById('diagnostics-summary').textContent.includes('关闭'));
} else {
 assert.ok(document.getElementById('diagnostics-notice').textContent.includes('无法读取'));
}
await document.getElementById('diagnostics-refresh').handlers.click();
assert.equal(calls,2);
console.log('Pages interactions passed');
"""
    path = tmp_path / "page-test.mjs"
    path.write_text(harness)
    result = subprocess.run(
        [node, str(path), str(root / "pages/status/diagnostics.js"), mode],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "Pages interactions passed" in result.stdout
    assert "innerHTML" not in script

// Native Page views. No secrets or operation originals enter browser persistence.
export function renderSources(ui) {
  const {main,el,card,field,button,jsonDetails,listen,run,call,refresh,sources,overview,diagnostics}=ui;
  main.append(card('独立来源，共享一个 Core / SELF','每组最多 8 个 entry，10 来源使用 8＋2；只核实当前组权限。新来源默认关闭。'));
  const switches=card('原始观察','开关只控制新采集与首次提交；学习、回复采用和图片额外解析不参与筛选。');
  const observe=el('input');observe.type='checkbox';observe.id='observation-enabled';observe.checked=!!sources.intents['observation.enabled'];
  const label=el('label','启用公开 handler 原始观察');label.htmlFor=observe.id;switches.append(label,observe);
  listen(observe,'change',()=>run(async()=>{await call('controls/save',{expected_revision:sources.revision,action_key:'observation.enabled',desired:observe.checked});await refresh();}));
  button(switches,'读取实际接入限额',async()=>{await call('ingress/limits',{});await refresh();});
  switches.append(jsonDetails('已核实配置与时间',sources.ingress_limits));main.append(switches);
  for(const group of sources.groups){
    const row=card(`授权组 ${group.group_id}`,`${group.entries.length} 个 entry · 版本 ${group.version} · ${group.current?'当前观测':'待核实／历史观测'}`);
    row.append(jsonDetails('权限与绑定',group));
    button(row,'核实此组权限',async()=>{await call('groups/check',{group_id:group.group_id});await refresh();});main.append(row);
  }
  const binding=el('form');binding.append(el('h3','绑定已有后端凭据'),el('p','更换组凭据会生成新绑定版本。旧未知项仍保留原绑定，不能自动改用新令牌。'));
  const groupId=field(binding,'group-id','授权组标识'),hostId=field(binding,'host-id','Core host 标识'),entries=field(binding,'entries','entry 列表（逗号分隔，最多 8 项）'),token=field(binding,'group-token','Host token','password');
  token.maxLength=4096;
  button(binding,'保存授权组',async()=>{const value={expected_revision:sources.revision,group_id:groupId.value,host_id:hostId.value,entries:entries.value.split(',').map(s=>s.trim()).filter(Boolean),token:token.value};token.value='';await call('groups/save',value);await refresh();});main.append(binding);
  for(const source of sources.sources){
    const row=card(`${source.kind==='group'?'群聊':'私聊'} ${source.conversation_id}`,`平台实例 ${source.platform_instance} · 机器人 ${source.bot_self} · entry ${source.entry_id} · 组 ${source.group_id}`);
    button(row,source.enabled?'停止此来源新接入':'启用此来源接入',async()=>{const value=Object.fromEntries(['platform_instance','bot_self','kind','conversation_id','entry_id','group_id'].map(k=>[k,source[k]]));value.enabled=!source.enabled;await call('sources/save',{expected_revision:sources.revision,source:value});await refresh();});main.append(row);
  }
  const sourceForm=el('form');sourceForm.append(el('h3','添加默认关闭的来源'));
  const sourceFields={};for(const [key,title] of [['platform_instance','AstrBot 平台实例 ID'],['bot_self','机器人 self_id'],['conversation_id','群号／私聊用户 ID'],['entry_id','独立 entry 标识'],['group_id','授权组标识']])sourceFields[key]=field(sourceForm,'source-'+key,title);
  const kindLabel=el('label','来源类型'),kind=el('select');kind.id='source-kind';kindLabel.htmlFor=kind.id;for(const [value,title] of [['group','群聊'],['private','私聊']]){const option=el('option',title);option.value=value;kind.append(option);}sourceForm.append(kindLabel,kind);
  button(sourceForm,'保存关闭的来源',async()=>{const source=Object.fromEntries(Object.entries(sourceFields).map(([k,v])=>[k,v.value]));source.kind=kind.value;source.enabled=false;await call('sources/save',{expected_revision:sources.revision,source});await refresh();});main.append(sourceForm);
  const registration=el('form');registration.append(el('h3','Core 来源登记'),el('p','需要本用户合法 Core 管理会话。登记结果与本地来源开关独立。'));
  const registrationFields={};for(const [key,title] of [['entry_id','entry 标识'],['host_id','host 标识'],['platform_id','Core 已配置平台 ID'],['external_entry_id','稳定外部来源身份']])registrationFields[key]=field(registration,'register-'+key,title);
  button(registration,'登记来源',async()=>{await call('ingress/register',{expected_revision:sources.revision,input:Object.fromEntries(Object.entries(registrationFields).map(([k,v])=>[k,v.value]))});await refresh();});
  button(registration,'读取登记列表',async()=>{const result=await call('ingress/list',{kind:'connections/hosts/list',after:''});if(registration.isConnected)registration.append(jsonDetails('Core 来源（第一页）',result));});main.append(registration);
  const issue=el('form');issue.append(el('h3','签发最小接入令牌'),el('p','只授予原始接入、媒体及原确认。秘密直接保存在后端。丢响应后先确认原操作；提交成功不等于秘密已取得。'));
  const issueGroup=field(issue,'issue-group','目标授权组'),issueHost=field(issue,'issue-host','Core host 标识'),issueEntries=field(issue,'issue-entries','entry 列表（最多 8 项）'),hours=field(issue,'issue-hours','有效小时','number','24');hours.min='1';hours.max='720';
  button(issue,'签发并绑定此组',async()=>{await call('ingress/token',{expected_revision:sources.revision,group_id:issueGroup.value,input:{host_id:issueHost.value,entries:issueEntries.value.split(',').map(x=>x.trim()).filter(Boolean),operations:['accept','confirm','media_upload','media_inspect'],expires_at_us:Math.floor((Date.now()+Number(hours.value)*3600000)*1000)}});await refresh();});
  button(issue,'读取令牌列表',async()=>{const result=await call('ingress/list',{kind:'tokens/list',after:''});if(!issue.isConnected)return;issue.append(jsonDetails('Core 令牌（第一页）',result));for(const token of result.items){if(token.state==='VALID')button(issue,`撤销 ${token.object_id.slice(0,12)}`,async()=>{await call('ingress/revoke',{expected_revision:overview.settings.revision,input:{token_id:token.object_id,expected_revision:token.revision}});await refresh();});}});main.append(issue);
  for(const op of diagnostics.operations.items.filter(x=>['source_register','token_create','token_revoke'].includes(x.kind))){const row=card('管理原操作',`${op.kind} · ${op.state} · 待清理 ${!!op.cleanup_pending}`);button(row,'合法原确认',async()=>{await call('ingress/confirm',{id:op.id});await refresh();});main.append(row);}
}

export function renderDelivery(ui){
  const {main,el,card,button,jsonDetails,call,refresh,sources,delivery,deliveryPage}=ui;
  main.append(card('有界交付与恢复','新消息和原确认分别计数。同来源未知项阻挡后续，关闭来源保留队列。没有删除未知项或换新 key 重试的入口。'));
  main.append(jsonDetails('实际限额与占用',delivery),jsonDetails('覆盖矩阵与无法实现保证',sources.coverage),jsonDetails('不可恢复缺口区间',delivery.gaps));
  if(delivery.storage_failed||delivery.memory_only_gaps||delivery.counts.gaps)main.append(card('容量／保存告警','存在接入缺口或持久化失败。只在内存记录的计数不保证崩溃后恢复。'));
  const pages=el('div',undefined,'actions');const prev=button(pages,'较新记录',()=>deliveryPage(Math.max(0,delivery.offset-50)));prev.disabled=delivery.offset===0;const next=button(pages,'较早记录',()=>deliveryPage(delivery.offset+50));next.disabled=delivery.offset+50>=Object.values(delivery.states).reduce((a,b)=>a+b,0);main.append(pages);
  const authority=el('select');authority.id='confirmation-group';const authorityLabel=el('label','重新授权原确认所用组');authorityLabel.htmlFor=authority.id;for(const g of sources.groups){const o=el('option',g.group_id);o.value=g.group_id;authority.append(o);}main.append(authorityLabel,authority,el('p','需要当前用户合法管理会话，重新核实同一实例、host、entry 与令牌权限。只确认已派发原操作，不授权首次提交。'));
  for(const row of delivery.items){const item=card(`接收序号 ${row.seq}`,`${row.state}${row.intake_paused?"（控制关闭，首次提交暂停）":""} · ${row.reason||'无额外原因'} · 首次提交 ${row.submitted} · 原确认 ${row.confirms} · 清理 ${row.cleanup_pending?'待完成':'无待清理观测'}`);item.append(el('small',row.source_id));if(row.state==='NOT_COMMITTED'&&!row.cleanup_pending)button(item,'确证未提交后以原输入续办',async()=>{await call('delivery/resume',{id:row.id});await refresh();});if(row.submitted&&(row.cleanup_pending||!['CONFIRMED','REJECTED','BLOCKED'].includes(row.state)))button(item,'调度原确认',async()=>{await call('delivery/confirm',{id:row.id});await refresh();});if(row.submitted&&(row.cleanup_pending||!['CONFIRMED','REJECTED','BLOCKED'].includes(row.state)))button(item,'重新授权后仅确认原操作',async()=>{await call('delivery/reauthorize-confirm',{id:row.id,group_id:authority.value,expected_revision:sources.revision});await refresh();});main.append(item);}
  main.append(card('页面退出边界','离页后停止新请求并丢弃迟到结果；父请求和远端操作不因此撤回。后台原操作跨刷新与进程恢复。'));
}

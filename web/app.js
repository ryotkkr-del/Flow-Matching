'use strict';
const $=id=>document.getElementById(id);
let route=null, previous=null, selected=0, frame=0, playing=false, last=0, cursor=0, busy=false, velocityScale=1, heatScale=1;
const scratch=document.createElement('canvas');scratch.width=scratch.height=28;const sx=scratch.getContext('2d');
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
function message(text,error=false){$('notice').textContent=text;$('notice').classList.toggle('error',error)}
function draw(id,values,mode='image',scale=1){
 const canvas=typeof id==='string'?$(id):id,ctx=canvas.getContext('2d'),pixels=sx.createImageData(28,28);
 for(let i=0;i<784;i++){
  let r=0,g=0,b=0,v=values?.[i]??-1;
  if(mode==='image'){r=g=b=Math.round(clamp((v+1)/2)*255)}
  else if(mode==='velocity'){const z=clamp(Math.abs(v)/scale);r=z*(v>=0?118:246);g=z*(v>=0?234:172);b=z*(v>=0?216:125)}
  else {const z=clamp(Math.abs(v)/scale);r=118*z;g=234*z;b=216*z}
  pixels.data.set([r,g,b,255],i*4);
 }
 sx.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=false;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(scratch,0,0,canvas.width,canvas.height);
}
function validate(d){
 if(d?.format!==1||d.method!=='euler'||!Number.isInteger(d.steps)||d.steps<1||d.steps>200||!Number.isInteger(d.count)||d.count<1||d.count>8)throw Error('対応していない生成経路です');
 if(!/^[a-f0-9]{64}$/.test(d.checkpoint_sha256)||!Number.isInteger(d.seed)||d.seed<0||d.seed>2147483647||!Number.isInteger(d.epoch)||d.epoch<1)throw Error('モデルの出所・学習情報がありません');
 if(!Array.isArray(d.times)||d.times.length!==d.steps+1||d.times.some((v,k)=>!Number.isFinite(v)||Math.abs(v-k/d.steps)>1e-7))throw Error('時刻情報が不正です');
 for(const [key,n] of [['states',d.steps+1],['velocities',d.steps]]){
  if(!Array.isArray(d[key])||d[key].length!==n)throw Error('軌道の長さが不正です');
  for(const row of d[key])if(!Array.isArray(row)||row.length!==d.count||row.some(a=>!Array.isArray(a)||a.length!==784||a.some(v=>typeof v!=='number'||!Number.isFinite(v))))throw Error('画像の形状または数値が不正です');
 }
 for(let k=0;k<d.steps;k++)for(let j=0;j<d.count;j++)for(let i=0;i<784;i++){
  const expected=d.states[k][j][i]+d.velocities[k][j][i]/d.steps;
  if(Math.abs(expected-d.states[k+1][j][i])>3e-5*Math.max(1,Math.abs(expected)))throw Error('オイラー更新と整合しない経路です');
 }
 return d;
}
function setRoute(d){
 route=validate(d);selected=0;frame=cursor=0;setPlaying(false);
 $('empty').hidden=true;$('time').disabled=false;$('time').max=d.steps;$('time').value=0;$('play').disabled=false;$('save').disabled=false;
 $('seed').value=d.seed;$('steps').value=[20,50,100].includes(d.steps)?d.steps:'50';$('count').value=d.count===1?'1':'4';
 velocityScale=1e-8;
 for(const row of d.velocities)for(const a of row)for(const v of a)velocityScale=Math.max(velocityScale,Math.abs(v));
 heatScale=velocityScale/d.steps;
 $('gallery').replaceChildren();$('gallery-section').hidden=d.count===1;
 for(let j=0;j<d.count;j++){
  const button=document.createElement('button'),c=document.createElement('canvas'),label=document.createElement('span');c.width=c.height=112;label.textContent=`NO. ${String(j+1).padStart(2,'0')}`;
  button.append(c,label);button.onclick=()=>{selected=j;render()};$('gallery').append(button);
 }
 $('provenance').textContent=`Euler · epoch ${d.epoch} · seed ${d.seed} · checkpoint SHA256 ${d.checkpoint_sha256}`;
 $('scale-note').textContent=`色の尺度（軌道全体で固定）：速度 ±${velocityScale.toFixed(3)} ／ 変化量 0〜${heatScale.toFixed(3)}。画像の明暗だけ表示時に[-1,1]へ制限。`;
 render();
}
function render(){
 if(!route)return;
 const d=route,k=Math.min(frame,d.steps-1),x=d.states[frame][selected],v=d.velocities[k][selected];
 draw('main',x);draw('final',d.states[d.steps][selected]);draw('current',d.states[k][selected]);draw('velocity',v,'velocity',velocityScale);draw('next',d.states[k+1][selected]);draw('heat',v.map(z=>Math.abs(z)/d.steps),'heat',heatScale);
 $('frame-label').textContent=`t = ${(frame/d.steps).toFixed(3)}`;$('step-label').textContent=`${frame} / ${d.steps}`;$('time').value=frame;$('seed-label').textContent=`SEED ${d.seed} / IMAGE ${selected+1}`;
 $('dt').textContent=`t = ${(k/d.steps).toFixed(3)} → ${((k+1)/d.steps).toFixed(3)} · Δt = ${(1/d.steps).toFixed(3)}`;
 $('stage-caption').textContent=frame===0?'ノイズからスタート':frame===d.steps?'ひとつの数字に、たどり着く。':'AIの指示で、少しずつ変わっていく。';
 Array.from($('gallery').children).forEach((b,j)=>{b.classList.toggle('active',j===selected);draw(b.querySelector('canvas'),d.states[frame][j])});
 if(previous){
  const oldFrame=Math.round(frame/d.steps*previous.steps);
  draw('old-comparison',previous.states[oldFrame][selected]);draw('new-comparison',x);
  $('compare-note').textContent=`同一の初期ノイズ・画像${selected+1}。旧 t=${(oldFrame/previous.steps).toFixed(3)} ／ 新 t=${(frame/d.steps).toFixed(3)}（各軌道の最寄りの計算時刻）。`;
 }
}
function setPlaying(value){playing=value;last=0;$('play').textContent=value?'Ⅱ':'▶';$('play').setAttribute('aria-label',value?'一時停止':'再生');$('play-state').textContent=route?(value?'実際の計算結果を再生しています':'スライダーで好きな瞬間へ'):'生成経路の読み込み待ち'}
function tick(time){
 if(playing&&route){if(last){cursor+=(time-last)/1000*route.steps/8*Number($('speed').value);const n=Math.min(route.steps,Math.floor(cursor));if(n!==frame){frame=n;render()}if(cursor>=route.steps)setPlaying(false)}last=time}
 requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
$('play').onclick=()=>{if(frame===route.steps){frame=cursor=0;render()}setPlaying(!playing)};
$('restart').onclick=()=>{if(route){frame=cursor=0;setPlaying(false);render()}};
$('time').oninput=()=>{setPlaying(false);frame=cursor=Number($('time').value);render()};
$('learn').onclick=()=>$('explain').showModal();$('close').onclick=()=>$('explain').close();
$('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(()=>message('このブラウザでは全画面表示が使えません',true))};
function setBusy(b){busy=b;$('busy').hidden=!b;$('generate').disabled=b;$('compare').disabled=b}
async function fetchJson(url,options){const r=await fetch(url,options);if(!r.ok){let detail;try{detail=(await r.json()).detail}catch{detail=r.statusText}throw Error(detail||`HTTP ${r.status}`)}return r.json()}
function comparison(old){
 previous=old;$('comparison-section').hidden=false;$('comparison').replaceChildren();
 for(const [id,steps] of [['old-comparison',old.steps],['new-comparison',route.steps]]){const box=document.createElement('div'),h=document.createElement('h3'),c=document.createElement('canvas');h.textContent=`${steps} STEPS`;c.id=id;c.width=c.height=224;box.append(h,c);$('comparison').append(box)}render();
}
$('close-compare').onclick=()=>{previous=null;$('comparison-section').hidden=true};
async function generate(compare=false){
 if(busy)return;
 const seed=Number($('seed').value),steps=Number($('steps').value),count=Number($('count').value);
 if(!Number.isInteger(seed)||seed<0||seed>2147483647){message('ノイズ番号は0〜2147483647の整数にしてください',true);return}
 const old=route;
 if(compare&&(!old||old.seed!==seed||old.count!==count)){message('比較には、表示中と同じノイズ番号・同時生成枚数を指定してください',true);return}
 if(compare&&old.steps===steps){message('計算ステップを変更してから比較してください',true);return}
 if(compare&&old.count>4){message('比較は4枚以下で行ってください',true);return}
 setBusy(true);message('実際のODE軌道を計算しています。しばらくお待ちください。');
 try{
  const j=await fetchJson('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed,steps,count})});
  const started=Date.now();let state=j.state;
  while(state!=='ready'){
   await new Promise(r=>setTimeout(r,400));const status=await fetchJson(`/api/jobs/${j.id}`);state=status.state;
   if(state==='error')throw Error(status.error);
   if(Date.now()-started>300000)throw Error('5分を超えました。計算は継続中の場合があります。同じ条件で再試行すると既存の処理を確認できます。');
  }
  const d=validate(await fetchJson(`/api/paths/${j.id}`));
  if(compare){if(d.checkpoint_sha256!==old.checkpoint_sha256)throw Error('モデルが異なるためステップ数だけの比較はできません');for(let j=0;j<count;j++)if(d.states[0][j].some((v,i)=>v!==old.states[0][j][i]))throw Error('初期ノイズが一致しません')}
  previous=null;$('comparison-section').hidden=true;setRoute(d);if(compare)comparison(old);message('生成完了。実際のODE計算の途中を自由にたどれます。');setPlaying(true);
 }catch(e){message(e.message,true)}finally{setBusy(false)}
}
$('generate').textContent='このノイズで生成 ↗';$('generate').onclick=()=>generate();$('compare').onclick=()=>generate(true);
const randomButton=document.createElement('button');randomButton.textContent='別のノイズへ ↻';randomButton.className='secondary';randomButton.onclick=()=>{$('seed').value=crypto.getRandomValues(new Uint32Array(1))[0]%2147483648;generate()};$('generate').after(randomButton);
$('empty-load').onclick=()=>$('import').click();
$('import').onchange=async e=>{
 const file=e.target.files[0];if(!file)return;
 try{
  if(file.size>80*1024*1024)throw Error('ファイルが大きすぎます（上限80MB）');
  let stream=file.stream();if(file.name.endsWith('.gz')){if(!window.DecompressionStream)throw Error('このブラウザではgzを開けません。JSONに解凍して読み込んでください');stream=stream.pipeThrough(new DecompressionStream('gzip'))}
  const reader=stream.getReader(),chunks=[];let size=0;
  while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>100*1024*1024){await reader.cancel();throw Error('展開後のサイズが100MBを超えています')}chunks.push(value)}
  const text=await new Blob(chunks).text();const d=validate(JSON.parse(text));previous=null;$('comparison-section').hidden=true;setRoute(d);
  message('保存した経路を再生中。Euler更新の整合を確認しました。出所は作成者のモデルとSHA256で別途確認してください。');
 }catch(e){message(e.message,true)}finally{e.target.value=''}
};
$('save').onclick=()=>{if(!route)return;const blob=new Blob([JSON.stringify(route)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`flow-seed${route.seed}-steps${route.steps}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)};
async function loadDemo(name){try{const d=validate(await fetchJson(`/api/demos/${encodeURIComponent(name)}`));previous=null;$('comparison-section').hidden=true;setRoute(d);message('保存された実モデルの生成経路を再生します。');setPlaying(true)}catch(e){message(e.message,true)}}
$('demos').onchange=e=>{if(e.target.value)loadDemo(e.target.value)};
async function init(){
 try{
  const s=await fetchJson('/api/status');
  for(const name of s.demos){const o=document.createElement('option');o.value=o.textContent=name;$('demos').append(o)}
  if(s.ready){message(`学習済みモデルを読み込みました（epoch ${s.epoch}）。ノイズから生成してみよう。`);generate()}
  else if(s.demos.length){message('保存した実モデルの経路を使うオフライン展示モードです。');await loadDemo(s.demos[0])}
  else message(s.error||'学習済みモデルが未配置です。READMEに沿って学習するか、実モデルから保存した経路を読み込んでください。',true);
  if(!s.ready){$('generate').disabled=true;$('compare').disabled=true;randomButton.disabled=true}
 }catch(e){message(`接続できません：${e.message}。経路ファイルの読み込みは利用できます。`,true)}
}
init();

'use strict';
const $=id=>document.getElementById(id);
let route=null, previous=null, selected=0, frame=0, playing=false, last=0, cursor=0, busy=false, velocityScale=1, heatScale=1;
let modelReady=false, demoNames=[], autoMode=false, autoTimer=null, loadVersion=0, autoIndex=-1;
let livePicked=0,livePreview=null,liveStep=0,liveSteps=0,liveSeed=0,liveGalleryReady=false,liveWaiting=false;
const scratch=document.createElement('canvas');scratch.width=scratch.height=28;const sx=scratch.getContext('2d');
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
function message(text,error=false){$('notice').textContent=text;$('notice').hidden=!error;$('notice').classList.toggle('error',error)}
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
// 推論の最中に到着した実フレーム。保存経路の再生とは区別して表示する。
function showLiveFrame(preview,step,steps,seed,waiting=false){
 if(!Array.isArray(preview)||!preview.length||preview.some(a=>!Array.isArray(a)||a.length!==784||a.some(v=>typeof v!=='number'||!Number.isFinite(v))))
  throw Error('途中画像の形式が不正です');
 livePreview=preview;liveStep=step;liveSteps=steps;liveSeed=seed;liveWaiting=waiting;
 $('empty').hidden=true;
 if(!liveGalleryReady){
  $('gallery').replaceChildren();
  for(let j=0;j<preview.length;j++){
   const button=document.createElement('button'),c=document.createElement('canvas'),label=document.createElement('span');
   c.width=c.height=112;label.textContent=`NO. ${String(j+1).padStart(2,'0')}`;
   button.append(c,label);
   button.onclick=()=>{livePicked=j;showLiveFrame(livePreview,liveStep,liveSteps,liveSeed,liveWaiting)};
   $('gallery').append(button);
  }
  liveGalleryReady=true;
 }
 $('gallery-section').hidden=preview.length===1;
 Array.from($('gallery').children).forEach((button,j)=>{
  button.classList.toggle('active',j===livePicked);
  draw(button.querySelector('canvas'),preview[j]);
 });
 draw('main',preview[livePicked]);
 $('frame-label').textContent=`t = ${(step/steps).toFixed(3)}`;
 $('seed-label').textContent=`SEED ${seed} / ${waiting?'NOISE':'LIVE'}`;
 $('stage-caption').textContent=waiting?'生成開始を押してください':'AIが実際に計算中';
 $('step-label').textContent=`${step} / ${steps}`;
 $('live-progress').hidden=waiting;
 $('live-progress').textContent=waiting?'':`● リアルタイム生成中  ${step} / ${steps} step`;
}
function validate(d){
 if(d.weights!==undefined&&!['raw','ema'].includes(d.weights))throw Error('重みの種類が不正です');
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
 $('live-progress').hidden=true;$('final').hidden=false;$('final-wait').hidden=true;
 $('internal-details').hidden=false;
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
 $('provenance').textContent=`Euler · epoch ${d.epoch} · ${d.weights==='ema'?'EMA weights':'raw weights'} · seed ${d.seed} · checkpoint SHA256 ${d.checkpoint_sha256}`;
 $('scale-note').textContent=`色の尺度（軌道全体で固定）：速度 ±${velocityScale.toFixed(3)} ／ 変化量 0〜${heatScale.toFixed(3)}。画像の明暗だけ表示時に[-1,1]へ制限。`;
 render();
}
function render(){
 if(!route)return;
 const d=route,k=Math.min(frame,d.steps-1),x=d.states[frame][selected],v=d.velocities[k][selected];
 draw('main',x);draw('final',d.states[d.steps][selected]);draw('current',d.states[k][selected]);draw('velocity',v,'velocity',velocityScale);draw('next',d.states[k+1][selected]);draw('heat',v.map(z=>Math.abs(z)/d.steps),'heat',heatScale);
 $('frame-label').textContent=`t = ${(frame/d.steps).toFixed(3)}`;$('step-label').textContent=`${frame} / ${d.steps}`;$('time').value=frame;$('seed-label').textContent=`SEED ${d.seed} / IMAGE ${selected+1}`;
 $('dt').textContent=`t = ${(k/d.steps).toFixed(3)} → ${((k+1)/d.steps).toFixed(3)} · Δt = ${(1/d.steps).toFixed(3)}`;
 $('stage-caption').textContent=frame===0?'ノイズ':frame===d.steps?'完了':'生成途中';
 Array.from($('gallery').children).forEach((b,j)=>{b.classList.toggle('active',j===selected);draw(b.querySelector('canvas'),d.states[frame][j])});
 if(previous){
  const oldFrame=Math.round(frame/d.steps*previous.steps);
  draw('old-comparison',previous.states[oldFrame][selected]);draw('new-comparison',x);
  $('compare-note').textContent=`同一の初期ノイズ・画像${selected+1}。旧 t=${(oldFrame/previous.steps).toFixed(3)} ／ 新 t=${(frame/d.steps).toFixed(3)}（各軌道の最寄りの計算時刻）。`;
 }
}
function setPlaying(value){playing=value;last=0;$('play').textContent=value?'Ⅱ':'▶';$('play').setAttribute('aria-label',value?'一時停止':'再生');$('play-state').textContent=route?(value?'再生中':'停止中'):'生成経路の読み込み待ち'}
function tick(time){
 if(playing&&route){if(last){cursor+=(time-last)/1000*route.steps/8*Number($('speed').value);const n=Math.min(route.steps,Math.floor(cursor));if(n!==frame){frame=n;render()}if(cursor>=route.steps){setPlaying(false);$('final').hidden=false;$('final-wait').hidden=true;if(autoMode)autoTimer=setTimeout(advanceAuto,1100)}}last=time}
 requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
$('play').onclick=()=>{stopAuto();if(frame===route.steps){frame=cursor=0;render()}setPlaying(!playing)};
$('restart').onclick=()=>{stopAuto();if(route){frame=cursor=0;setPlaying(false);render()}};
$('time').oninput=()=>{stopAuto();setPlaying(false);frame=cursor=Number($('time').value);render();if(frame===route.steps){$('final').hidden=false;$('final-wait').hidden=true}};
$('learn').onclick=()=>$('explain').showModal();$('close').onclick=()=>$('explain').close();
$('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(()=>message('このブラウザでは全画面表示が使えません',true))};
function setBusy(b){
 busy=b;$('busy').hidden=!b;
 const unavailable=!modelReady&&!demoNames.length;
 $('generate').disabled=b||unavailable;$('compare').disabled=b||unavailable;
 randomButton.disabled=b||unavailable;
}
function stopAuto(){if(autoMode)loadVersion++;autoMode=false;clearTimeout(autoTimer);$('auto').textContent='自動展示';$('auto').setAttribute('aria-pressed','false')}
function autoNames(){const preferred=demoNames.filter(n=>/-steps50\.json\.gz$/.test(n));return preferred.length?preferred:demoNames}
async function advanceAuto(){
 if(!autoMode)return;
 const names=autoNames();
 if(names.length){autoIndex=(autoIndex+1)%names.length;await loadDemo(names[autoIndex],true)}
 else if(route){frame=cursor=0;render();setPlaying(true)}
}
$('auto').onclick=()=>{
 if(autoMode){stopAuto();setPlaying(false);return}
 if(!route&&!demoNames.length){message('先に生成経路を読み込んでください',true);return}
 autoIndex=route?autoNames().indexOf(`seed${route.seed}-steps${route.steps}.json.gz`):-1;
 autoMode=true;$('auto').textContent='自動展示 ON';$('auto').setAttribute('aria-pressed','true');
 if(route){if(frame===route.steps){frame=cursor=0;render()}setPlaying(true)}else advanceAuto();
};
async function fetchJson(url,options){const r=await fetch(url,options);if(!r.ok){let detail;try{detail=(await r.json()).detail}catch{detail=r.statusText}throw Error(detail||`HTTP ${r.status}`)}return r.json()}
// Present the exact x0 used by the sampler, but do not start model inference.
async function prepareNoise(){
 if(!modelReady||busy)return;
 const seed=Number($('seed').value),count=Number($('count').value);
 if($('seed').value===''||!Number.isInteger(seed)||seed<0||seed>2147483647){
  message('ノイズ番号は0〜2147483647の整数にしてください',true);$('generate').disabled=true;return;
 }
 stopAuto();setPlaying(false);
 const version=++loadVersion;
 $('generate').disabled=true;
 try{
  const data=await fetchJson(`/api/noise?seed=${seed}&count=${count}`);
  if(version!==loadVersion)return;
  if(data.seed!==seed||data.count!==count)throw Error('表示するノイズの条件が一致しません');
  route=null;previous=null;livePicked=0;liveGalleryReady=false;
  $('comparison-section').hidden=true;
  showLiveFrame(data.pixels,0,Number($('steps').value),seed,true);
  $('final').hidden=true;$('final-wait').hidden=false;$('final-wait').textContent='生成開始後に表示します';
  $('internal-details').hidden=true;$('scale-note').hidden=true;
  $('time').disabled=true;$('play').disabled=true;$('restart').disabled=true;
  $('save').disabled=true;$('compare').disabled=true;$('auto').disabled=true;
  $('play-state').textContent='「生成開始」を押してください';
  message('ノイズを用意しました。「生成開始」を押すと、このノイズから数字を作ります。');
 }catch(e){
  if(version===loadVersion){$('empty').hidden=false;message(`ノイズを表示できません：${e.message}`,true)}
 }finally{
  if(version===loadVersion)$('generate').disabled=!livePreview||!liveWaiting;
 }
}
$('seed').onchange=()=>prepareNoise();
$('count').onchange=()=>prepareNoise();
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
 stopAuto();const version=++loadVersion;setPlaying(false);setBusy(true);
 if(modelReady){
  $('play').disabled=true;$('restart').disabled=true;$('time').disabled=true;$('auto').disabled=true;
  $('final').hidden=true;$('final-wait').hidden=false;$('final-wait').textContent='AIが最終画像を計算しています…';
  $('live-progress').hidden=false;$('live-progress').textContent='AIの生成を開始しています…';
  livePicked=0;liveGalleryReady=false;livePreview=null;
  $('gallery-section').hidden=true;
 }
 message(modelReady?'学習済みAIが今、ノイズから画像を生成しています。':'保存済みの実生成経路を読み込んでいます。');
 try{
  let d;
  if(!modelReady){
   const name=`seed${seed}-steps${steps}.json.gz`;
   if(!demoNames.includes(name))throw Error('この条件の保存デモはありません。画面下部の保存済みデモから選んでください。');
   d=validate(await fetchJson(`/api/demos/${encodeURIComponent(name)}`));
   if(d.count!==count)throw Error(`このデモは${d.count}枚です。同時生成の枚数を合わせてください。`);
  }else{
   const j=await fetchJson('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed,steps,count})});
   const started=Date.now();let state=j.state,seen=-1;
   while(state!=='ready'){
    await new Promise(r=>setTimeout(r,160));
    if(version!==loadVersion)return;
    const status=await fetchJson(`/api/jobs/${encodeURIComponent(j.id)}?after=${seen}`);
    state=status.state;
    if(state==='error')throw Error(status.error);
    if(state==='running'&&Array.isArray(status.preview)&&Number.isInteger(status.step)&&status.step>seen){
     showLiveFrame(status.preview,status.step,steps,seed);
     seen=status.step;
    }
    if(Date.now()-started>300000)throw Error('5分を超えました。計算は継続中の場合があります。同じ条件で再試行すると既存の処理を確認できます。');
   }
   d=validate(await fetchJson(`/api/paths/${j.id}`));
  }
  if(version!==loadVersion)return;
  if(compare){if(d.checkpoint_sha256!==old.checkpoint_sha256||(d.weights||'raw')!==(old.weights||'raw'))throw Error('モデルが異なるためステップ数だけの比較はできません');for(let j=0;j<count;j++)if(d.states[0][j].some((v,i)=>v!==old.states[0][j][i]))throw Error('初期ノイズが一致しません')}
  previous=null;$('comparison-section').hidden=true;setRoute(d);if(compare)comparison(old);
  message(modelReady?'生成完了。実際のODE計算の途中を自由にたどれます。':'保存デモを再生中。実モデルが計算した経路を自由にたどれます。');
  if(modelReady){frame=cursor=d.steps;render();setPlaying(false)}
  else {
   $('final').hidden=true;$('final-wait').hidden=false;$('final-wait').textContent='再生が終わると表示されます';
   setPlaying(true);
  }
 }catch(e){
  if(version===loadVersion){
   message(e.message,true);
   if(route){setRoute(route);render()}else $('empty').hidden=false;
  }
 }finally{
  if(version===loadVersion){
   $('live-progress').hidden=true;
   if(modelReady){$('final').hidden=false;$('final-wait').hidden=true}
   $('play').disabled=!route;$('restart').disabled=!route;$('time').disabled=!route;$('auto').disabled=false;
  }
  setBusy(false);
 }
}
$('generate').textContent='生成開始 ↗';$('generate').onclick=()=>generate();$('compare').onclick=()=>generate(true);
const randomButton=document.createElement('button');randomButton.textContent='別のノイズへ ↻';randomButton.className='secondary';randomButton.onclick=()=>{
 if(busy)return;
 if(modelReady)$('seed').value=crypto.getRandomValues(new Uint32Array(1))[0]%2147483648;
 else{
  const candidates=demoNames.map(n=>/^seed(\d+)-steps(\d+)\.json\.gz$/.exec(n)).filter(m=>m&&Number(m[2])===Number($('steps').value)&&Number(m[1])!==Number($('seed').value));
  if(!candidates.length){message('このステップ数では、別のノイズの保存デモがありません',true);return}
  $('seed').value=candidates[crypto.getRandomValues(new Uint32Array(1))[0]%candidates.length][1];
 }
 if(modelReady)prepareNoise();
 else loadDemo(`seed${$('seed').value}-steps${$('steps').value}.json.gz`,false,false);
};$('generate').after(randomButton);
$('empty-load').onclick=()=>$('import').click();
$('import').onchange=async e=>{
 const file=e.target.files[0];if(!file)return;stopAuto();const version=++loadVersion;
 try{
  if(file.size>80*1024*1024)throw Error('ファイルが大きすぎます（上限80MB）');
  let stream=file.stream();if(file.name.endsWith('.gz')){if(!window.DecompressionStream)throw Error('このブラウザではgzを開けません。JSONに解凍して読み込んでください');stream=stream.pipeThrough(new DecompressionStream('gzip'))}
  const reader=stream.getReader(),chunks=[];let size=0;
  while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>100*1024*1024){await reader.cancel();throw Error('展開後のサイズが100MBを超えています')}chunks.push(value)}
  const text=await new Blob(chunks).text();const d=validate(JSON.parse(text));if(version!==loadVersion)return;previous=null;$('comparison-section').hidden=true;setRoute(d);
  message('保存した経路を再生中。Euler更新の整合を確認しました。出所は作成者のモデルとSHA256で別途確認してください。');
 }catch(e){message(e.message,true)}finally{e.target.value=''}
};
$('save').onclick=()=>{if(!route)return;const blob=new Blob([JSON.stringify(route)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`flow-seed${route.seed}-steps${route.steps}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)};
async function loadDemo(name,automatic=false,playImmediately=true){
 if(!automatic)stopAuto();
 const version=++loadVersion;
 try{
  const d=validate(await fetchJson(`/api/demos/${encodeURIComponent(name)}`));
  if(version!==loadVersion||(automatic&&!autoMode))return;
  previous=null;$('comparison-section').hidden=true;setRoute(d);$('demos').value=name;
  if(!playImmediately){
   $('final').hidden=true;$('final-wait').hidden=false;$('final-wait').textContent='再生開始後に表示します';
   $('stage-caption').textContent='再生開始を押してください';
   $('play-state').textContent='保存デモ：再生待ち';
  }
  message(automatic?'自動展示：保存した実生成経路を巡回しています。再生ボタンやスライダーで自動展示を終了できます。':playImmediately?'保存された実モデルの生成経路を再生します。':'保存デモの最初のノイズを表示しています。「保存デモを再生」を押してください。');
  if(playImmediately)setPlaying(true);
 }catch(e){if(version===loadVersion){stopAuto();message(e.message,true)}}
}
$('demos').onchange=e=>{if(e.target.value)loadDemo(e.target.value)};
async function init(){
 try{
  const s=await fetchJson('/api/status');modelReady=Boolean(s.ready);demoNames=Array.isArray(s.demos)?s.demos.filter(n=>typeof n==='string'):[];
  for(const name of demoNames){const o=document.createElement('option');o.value=o.textContent=name;$('demos').append(o)}
  $('generate').textContent=modelReady?'生成開始 ↗':'保存デモを再生 ↗';
  setBusy(false);
  if(modelReady){message(`学習済みモデルを読み込みました（epoch ${s.epoch}）。初期ノイズを準備しています。`);await prepareNoise()}
  else if(demoNames.length){message('保存デモの初期ノイズを準備しています。');await loadDemo(demoNames.find(n=>/-steps50\.json\.gz$/.test(n))||demoNames[0],false,false)}
  else message(s.error||'学習済みモデルが未配置です。READMEに沿って学習するか、実モデルから保存した経路を読み込んでください。',true);
 }catch(e){setBusy(false);message(`接続できません：${e.message}。経路ファイルの読み込みは利用できます。`,true)}
}
init();

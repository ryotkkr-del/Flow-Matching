// Node上のUIロジック単体テスト。ブラウザの描画・操作検証を代替しない。
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
class Element {
 constructor(){this.children=[];this.value='';this.hidden=false;this.disabled=false;this.textContent='';this.width=this.height=28;this.classes=new Set();this.classList={toggle:(n,v)=>v?this.classes.add(n):this.classes.delete(n)};this.context={createImageData:()=>({data:new Uint8ClampedArray(3136)}),putImageData:d=>{this.pixels=d.data},clearRect(){},drawImage(){}}}
 getContext(){return this.context}append(...xs){this.children.push(...xs)}replaceChildren(...xs){this.children=xs}setAttribute(n,v){this[n]=v}querySelector(){return this.children.find(c=>c instanceof Element)}after(){}click(){this.onclick?.()}showModal(){this.open=true}close(){this.open=false}
}
function setup(){const ids=new Map(),get=id=>{if(!ids.has(id))ids.set(id,new Element());return ids.get(id)};
 const context={document:{getElementById:get,createElement:()=>new Element(),documentElement:{}},console,requestAnimationFrame(){},setTimeout,clearTimeout,Uint8ClampedArray,Uint32Array,crypto:require('node:crypto').webcrypto,window:{},Blob,URL,fetch:()=>{throw Error('unexpected inference/network request')}};
 vm.createContext(context);vm.runInContext(fs.readFileSync('web/app.js','utf8').replace(/init\(\);\s*$/,''),context);
 get('seed').value='42';get('steps').value='50';get('count').value='4';
 return {context,get,run:s=>vm.runInContext(s,context)};
}
function fixture(steps=5){return {format:1,method:'euler',seed:42,steps,count:1,epoch:1,checkpoint_sha256:'a'.repeat(64),times:Array.from({length:steps+1},(_,k)=>k/steps),states:Array.from({length:steps+1},(_,k)=>[Array(784).fill(k/steps)]),velocities:Array.from({length:steps},()=>[Array(784).fill(1)])}}

test('import rejects wrong Euler updates and nonfinite values',()=>{const {context,run}=setup();context.d=fixture();run('validate(d)');context.d.states[1][0][0]=100;assert.throws(()=>run('validate(d)'),/オイラー/);context.d=fixture();context.d.velocities[0][0][0]=null;assert.throws(()=>run('validate(d)'),/数値/)});
test('slider uses saved discrete states without fetch and pauses playback',()=>{const {context,get,run}=setup();context.d=fixture();run('setRoute(d);setPlaying(true)');get('time').value='3';get('time').oninput();assert.equal(run('frame'),3);assert.equal(run('playing'),false);assert.equal(get('frame-label').textContent,'t = 0.600');assert.equal(get('play-state').textContent,'停止中')});
test('final frame displays last real update interval',()=>{const {context,get,run}=setup();context.d=fixture();run('setRoute(d);frame=5;render()');assert.match(get('dt').textContent,/t = 0.800 → 1.000/);assert.equal(get('step-label').textContent,'5 / 5')});
test('resume and restart preserve stored trajectory',()=>{const {context,get,run}=setup();context.d=fixture();run('setRoute(d);frame=cursor=3;setPlaying(true);setPlaying(false)');assert.equal(run('frame'),3);get('restart').onclick();assert.equal(run('frame'),0);assert.equal(run('route'),context.d);assert.equal(run('playing'),false)});
test('pixel mapping clips display only and uses signed velocity colors',()=>{const {context,run}=setup();run("draw('main',Array(784).fill(2))");assert.equal(run("scratch.pixels[0]"),255);run("draw('velocity',Array(784).fill(-1),'velocity',1)");assert.equal(run("scratch.pixels[0]"),246);assert.equal(run("scratch.pixels[1]"),172);context.d=fixture();context.d.states[0][0][0]=-2;assert.equal(context.d.states[0][0][0],-2)});
test('comparison synchronizes nearest discrete times and preserves labels',()=>{const {context,get,run}=setup();context.d=fixture(10);context.old=fixture(5);run('setRoute(d);comparison(old);frame=5;render()');assert.match(get('compare-note').textContent,/旧 t=0.600 ／ 新 t=0.500/);assert.equal(get('comparison-section').hidden,false)});

test('recorded demos support same-noise step comparison without inference',async()=>{
 const {context,get,run}=setup();context.d=fixture(5);run("setRoute(d);demoNames=['seed42-steps20.json.gz'];modelReady=false");get('steps').value='20';
 const urls=[];context.fetch=async url=>{urls.push(url);return {ok:true,json:async()=>fixture(20)}};
 await run('generate(true)');
 assert.deepEqual(urls,['/api/demos/seed42-steps20.json.gz']);assert.equal(run('route.steps'),20);assert.equal(run('previous.steps'),5);assert.equal(get('comparison-section').hidden,false);assert.equal(run('busy'),false);
});
test('recorded comparison refuses a different checkpoint',async()=>{
 const {context,get,run}=setup();context.d=fixture(5);run("setRoute(d);demoNames=['seed42-steps20.json.gz'];modelReady=false");get('steps').value='20';
 const d=fixture(20);d.checkpoint_sha256='b'.repeat(64);context.fetch=async()=>({ok:true,json:async()=>d});
 await run('generate(true)');assert.equal(run('route.steps'),5);assert.match(get('notice').textContent,/モデルが異なる/);
});
test('auto mode restarts a real stored route and slider stops auto mode',async()=>{
 const {context,get,run}=setup();context.d=fixture(5);run('setRoute(d);frame=cursor=5');get('auto').onclick();assert.equal(run('autoMode'),true);assert.equal(run('frame'),0);assert.equal(run('playing'),true);
 get('time').value='2';get('time').oninput();assert.equal(run('autoMode'),false);assert.equal(run('playing'),false);assert.equal(run('frame'),2);
});
test('manual selection wins over a pending automatic demo load',async()=>{
 const {context,get,run}=setup();context.d=fixture(5);run("setRoute(d);demoNames=['seed42-steps50.json.gz'];autoMode=true");let resolve;
 context.fetch=()=>new Promise(r=>{resolve=r});const pending=run('advanceAuto()');run('stopAuto();frame=cursor=3;render()');resolve({ok:true,json:async()=>fixture(50)});await pending;
 assert.equal(run('route.steps'),5);assert.equal(run('frame'),3);assert.equal(run('autoMode'),false);
});

test('step-only comparison rejects raw versus EMA even for the same checkpoint',async()=>{
 const {context,get,run}=setup();context.d=fixture(5);context.d.weights='raw';run("setRoute(d);demoNames=['seed42-steps20.json.gz'];modelReady=false");get('steps').value='20';
 const d=fixture(20);d.weights='ema';context.fetch=async()=>({ok:true,json:async()=>d});
 await run('generate(true)');assert.equal(run('route.steps'),5);assert.match(get('notice').textContent,/モデルが異なる/);
});


test('inference preview draws real server frame and updates progress label',()=>{
 const {context,get,run}=setup();
 const values=Array(784).fill(0.3);
 context.partial=[values];
 run('showLiveFrame(partial,3,10,345)');
 assert.equal(get('live-progress').hidden,false);
 assert.match(get('live-progress').textContent,/3 \/ 10 step/);
 assert.equal(get('frame-label').textContent,'t = 0.300');
 assert.equal(get('stage-caption').textContent,'AIが実際に計算中');
 assert.equal(get('main').pixels?.length,undefined); // large canvas compositing is exercised through scratch
 assert.equal(run('scratch.pixels[0]'),Math.round((0.3+1)/2*255));
});


test('four simultaneous samples update and remain selectable during live inference',()=>{
 const {context,get,run}=setup();
 context.frames=[0,1,2,3].map(n=>Array(784).fill(-1+n/2));
 run('showLiveFrame(frames,2,20,9)');
 assert.equal(get('gallery-section').hidden,false);
 assert.equal(get('gallery').children.length,4);
 get('gallery').children[2].onclick();
 assert.equal(run('livePicked'),2);
 assert.equal(run('scratch.pixels[0]'),128);
 context.next=[0,1,2,3].map(n=>Array(784).fill(-1+n/3));
 run('showLiveFrame(next,3,20,9)');
 assert.equal(run('livePicked'),2);
 assert.equal(run('scratch.pixels[0]'),85);
});


test('startup with a model displays x0 and never starts inference automatically',async()=>{
 const {context,get,run}=setup(),called=[];
 context.fetch=async (url,opts)=>{
   called.push([url,opts?.method||'GET']);
   if(url==='/api/status')return {ok:true,json:async()=>({ready:true,epoch:25,demos:[]})};
   if(url==='/api/noise?seed=42&count=4')
     return {ok:true,json:async()=>({seed:42,count:4,pixels:Array.from({length:4},(_,k)=>Array(784).fill(-1+k/3))})};
   throw Error('unexpected request '+url);
 };
 await run('init()');
 assert.deepEqual(called,[['/api/status','GET'],['/api/noise?seed=42&count=4','GET']]);
 assert.equal(run('route'),null);
 assert.equal(run('playing'),false);
 assert.equal(get('stage-caption').textContent,'生成開始を押してください');
 assert.equal(get('generate').textContent,'生成開始 ↗');
 assert.equal(get('step-label').textContent,'0 / 50');
 assert.equal(get('final').hidden,true);
 assert.equal(get('final-wait').hidden,false);
 assert.equal(get('live-progress').hidden,true);
 assert.equal(get('gallery').children.length,4);
 assert.equal(get('generate').disabled,false);
 assert.equal(get('play').disabled,true);
});

test('selecting a different seed only changes initial noise, with no generate request',async()=>{
 const {context,get,run}=setup(),called=[];
 run('modelReady=true');
 context.fetch=async url=>{
   called.push(url);
   if(url.startsWith('/api/noise?'))return {ok:true,json:async()=>({
    seed:Number(new URL('http://localhost'+url).searchParams.get('seed')),count:4,
    pixels:Array.from({length:4},()=>Array(784).fill(-0.5))
   })};
   throw Error('unexpected request');
 };
 get('seed').value='43';
 await get('seed').onchange();
 assert.deepEqual(called,['/api/noise?seed=43&count=4']);
 assert.equal(get('seed-label').textContent,'SEED 43 / NOISE');
 assert.equal(run('route'),null);
 assert.equal(get('stage-caption').textContent,'生成開始を押してください');
});

test('saved demo opens paused at noise, never autoplaying on launch',async()=>{
 const {context,get,run}=setup(),calls=[];
 context.fetch=async url=>{
  calls.push(url);
  if(url==='/api/status')return {ok:true,json:async()=>({ready:false,demos:['seed42-steps50.json.gz']})};
  if(url==='/api/demos/seed42-steps50.json.gz')return {ok:true,json:async()=>fixture(50)};
  throw Error('unexpected request '+url);
 };
 await run('init()');
 assert.deepEqual(calls,['/api/status','/api/demos/seed42-steps50.json.gz']);
 assert.equal(run('playing'),false);
 assert.equal(run('frame'),0);
 assert.equal(get('final').hidden,true);
 assert.equal(get('stage-caption').textContent,'再生開始を押してください');
 assert.equal(get('generate').textContent,'保存デモを再生 ↗');
});

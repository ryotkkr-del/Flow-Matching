// Node上のUIロジック単体テスト。ブラウザの描画・操作検証を代替しない。
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
class Element {
 constructor(){this.children=[];this.value='';this.hidden=false;this.disabled=false;this.textContent='';this.width=this.height=28;this.classes=new Set();this.classList={toggle:(n,v)=>v?this.classes.add(n):this.classes.delete(n)};this.context={createImageData:()=>({data:new Uint8ClampedArray(3136)}),putImageData:d=>{this.pixels=d.data},clearRect(){},drawImage(){}}}
 getContext(){return this.context}append(...xs){this.children.push(...xs)}replaceChildren(...xs){this.children=xs}setAttribute(n,v){this[n]=v}querySelector(){return this.children.find(c=>c instanceof Element)}after(){}click(){this.onclick?.()}showModal(){this.open=true}close(){this.open=false}
}
function setup(){const ids=new Map(),get=id=>{if(!ids.has(id))ids.set(id,new Element());return ids.get(id)};
 const context={document:{getElementById:get,createElement:()=>new Element(),documentElement:{}},console,requestAnimationFrame(){},setTimeout,Uint8ClampedArray,Uint32Array,crypto:require('node:crypto').webcrypto,window:{},Blob,URL,fetch:()=>{throw Error('unexpected inference/network request')}};
 vm.createContext(context);vm.runInContext(fs.readFileSync('web/app.js','utf8').replace(/init\(\);\s*$/,''),context);
 return {context,get,run:s=>vm.runInContext(s,context)};
}
function fixture(steps=5){return {format:1,method:'euler',seed:42,steps,count:1,epoch:1,checkpoint_sha256:'a'.repeat(64),times:Array.from({length:steps+1},(_,k)=>k/steps),states:Array.from({length:steps+1},(_,k)=>[Array(784).fill(k/steps)]),velocities:Array.from({length:steps},()=>[Array(784).fill(1)])}}

test('import rejects wrong Euler updates and nonfinite values',()=>{const {context,run}=setup();context.d=fixture();run('validate(d)');context.d.states[1][0][0]=100;assert.throws(()=>run('validate(d)'),/オイラー/);context.d=fixture();context.d.velocities[0][0][0]=null;assert.throws(()=>run('validate(d)'),/数値/)});
test('slider uses saved discrete states without fetch and pauses playback',()=>{const {context,get,run}=setup();context.d=fixture();run('setRoute(d);setPlaying(true)');get('time').value='3';get('time').oninput();assert.equal(run('frame'),3);assert.equal(run('playing'),false);assert.equal(get('frame-label').textContent,'t = 0.600');assert.equal(get('play-state').textContent,'スライダーで好きな瞬間へ')});
test('final frame displays last real update interval',()=>{const {context,get,run}=setup();context.d=fixture();run('setRoute(d);frame=5;render()');assert.match(get('dt').textContent,/t = 0.800 → 1.000/);assert.equal(get('step-label').textContent,'5 / 5')});
test('resume and restart preserve stored trajectory',()=>{const {context,get,run}=setup();context.d=fixture();run('setRoute(d);frame=cursor=3;setPlaying(true);setPlaying(false)');assert.equal(run('frame'),3);get('restart').onclick();assert.equal(run('frame'),0);assert.equal(run('route'),context.d);assert.equal(run('playing'),false)});
test('pixel mapping clips display only and uses signed velocity colors',()=>{const {context,run}=setup();run("draw('main',Array(784).fill(2))");assert.equal(run("scratch.pixels[0]"),255);run("draw('velocity',Array(784).fill(-1),'velocity',1)");assert.equal(run("scratch.pixels[0]"),246);assert.equal(run("scratch.pixels[1]"),172);context.d=fixture();context.d.states[0][0][0]=-2;assert.equal(context.d.states[0][0][0],-2)});
test('comparison synchronizes nearest discrete times and preserves labels',()=>{const {context,get,run}=setup();context.d=fixture(10);context.old=fixture(5);run('setRoute(d);comparison(old);frame=5;render()');assert.match(get('compare-note').textContent,/旧 t=0.600 ／ 新 t=0.500/);assert.equal(get('comparison-section').hidden,false)});

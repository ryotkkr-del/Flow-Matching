// UI review only. The offline Python inference server is server.py.
const http=require('node:http'),fs=require('node:fs'),path=require('node:path');
http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost');let file;
 if(url.pathname==='/api/status'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({ready:false,demos:fs.readdirSync('demos').filter(n=>n.endsWith('.json.gz'))}));return}
 if(url.pathname.startsWith('/api/demos/')){const name=decodeURIComponent(url.pathname.split('/').pop());if(!fs.readdirSync('demos').includes(name)){res.writeHead(404);res.end();return}file=path.join('demos',name);res.setHeader('Content-Encoding','gzip');res.setHeader('Content-Type','application/json')}
 else if(url.pathname==='/')file='web/index.html';else if(/^\/web\/[a-zA-Z0-9_.-]+$/.test(url.pathname))file=url.pathname.slice(1);else{res.writeHead(404);res.end();return}
 if(!fs.existsSync(file)){res.writeHead(404);res.end();return}
 const ext=path.extname(file);if(ext==='.js')res.setHeader('Content-Type','text/javascript');if(ext==='.css')res.setHeader('Content-Type','text/css');if(ext==='.html')res.setHeader('Content-Type','text/html');fs.createReadStream(file).pipe(res);
}).listen(4173,'0.0.0.0');

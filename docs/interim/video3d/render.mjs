// Renders video.html to a silent 1920 x 1080 MP4.
//   node render.mjs                     the whole video to build/d2_toy3d.mp4
//   node render.mjs --still 95 [...]    one PNG per time, to build/still_95.png
//   node render.mjs --from 60 --to 85   only that span, to build/part.mp4
// Needs google-chrome, ffmpeg and network access (Three.js and the fonts come from CDNs).
// No npm packages: it drives Chrome over the DevTools protocol with Node's built-in
// WebSocket, so it runs on Node 22+. Each frame is drawn by seek(t) in the page,
// never by wall-clock time, so a slow machine gives the same video, only later.
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const build = join(here, 'build');
const FPS = 30;

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
const stills = args.includes('--still') ? args.slice(args.indexOf('--still') + 1).filter(a => !a.startsWith('--')).map(Number) : null;
mkdirSync(build, { recursive: true });

// Chrome with a DevTools port. Unlike the 2D video this one needs WebGL, so the GPU
// stays on; ANGLE falls back to SwiftShader where there is none, only slower.
const port = 9300 + Math.floor(Math.random() * 500);
// Outside the repo: the profile holds symlinks, and a test that walks the repo trips on them.
const profile = mkdtempSync(join(tmpdir(), 'd2-chrome-'));
const chrome = spawn('google-chrome', ['--headless=new', '--no-sandbox', '--hide-scrollbars', '--ignore-gpu-blocklist',
  // Plain --use-angle=vulkan fails headless; it needs these features and no Vulkan surface.
  '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--disable-vulkan-surface', '--disable-gpu-compositing', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--allow-file-access-from-files',
  '--window-size=1920,1080', 'about:blank'], { stdio: 'ignore' });
let target;
for (let i = 0; i < 100 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === 'page'); }
  catch { await new Promise(r => setTimeout(r, 100)); }
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r, { once: true }));
let nextId = 1;
const waiting = new Map();
ws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  // A failed CDN import or WebGL context otherwise shows only as a page that never gets seek().
  if (m.method === 'Runtime.exceptionThrown') console.error('page:', m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') console.error('page:', m.params.args.map(a => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((res, rej) => {
  const id = nextId++;
  waiting.set(id, m => (m.error ? rej(new Error(`${method}: ${m.error.message}`)) : res(m.result)));
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async expr => {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || expr);
  return r.result.value;
};

await send('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send('Page.enable');
await send('Runtime.enable');
await send('Page.navigate', { url: `file://${join(here, 'video.html')}?t=0` });
// The page builds its scenes only after the CDN modules and fonts load.
for (let i = 0; i < 300 && (await evaluate('typeof window.seek')) !== 'function'; i++) await new Promise(r => setTimeout(r, 100));
const total = await evaluate('window.TOTAL');
console.log('renderer:', await evaluate(`(() => { const g = document.createElement('canvas').getContext('webgl2'); const e = g.getExtension('WEBGL_debug_renderer_info'); return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; })()`));
const shot = async (format, quality) => Buffer.from((await send('Page.captureScreenshot', { format, quality })).data, 'base64');

try {
  if (stills) {
    for (const t of stills) {
      await evaluate(`seek(${t})`);
      const out = join(build, `still_${t}.png`);
      writeFileSync(out, await shot('png'));
      console.log(out);
    }
  } else {
    const from = +opt('--from', 0), to = Math.min(+opt('--to', total), total);
    const out = join(build, args.includes('--from') || args.includes('--to') ? 'part.mp4' : 'd2_toy3d.mp4');
    const ff = spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
      // Same encoder settings as the 2D video: tagged BT.709, a keyframe every second,
      // no B-frames, so VLC seeks cleanly.
      '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-profile:v', 'high', '-level', '4.1',
      '-g', String(FPS), '-keyint_min', String(FPS), '-bf', '0', '-x264-params', 'weightp=0',
      '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
      '-an', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
    const n = Math.round((to - from) * FPS);
    for (let i = 0; i < n; i++) {
      await evaluate(`seek(${from + i / FPS})`);
      const buf = await shot('jpeg', 95);
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      if (i % FPS === 0) process.stdout.write(`\r${(from + i / FPS).toFixed(0)} / ${to} s`);
    }
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
    console.log(`\n${out}`);
  }
} finally {
  ws.close();
  // Chrome writes to its profile until it exits, so delete the profile after that.
  await new Promise(r => { chrome.once('exit', r); chrome.kill(); });
  rmSync(profile, { recursive: true, force: true, maxRetries: 5 });
}

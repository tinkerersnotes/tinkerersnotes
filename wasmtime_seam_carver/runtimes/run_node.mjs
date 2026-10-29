// Times seam_carver.wasm in Node, whose WebAssembly engine is Chrome's V8. Reads the pixels
// make_page.py wrote to build/tundra.rgb. Usage: node run_node.mjs
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = (name) => fileURLToPath(new URL(name, import.meta.url));
const { compare } = createRequire(import.meta.url)('./host.js');
const wasm = readFileSync(here('../seam_carver.wasm'));
const rgb = new Uint8Array(readFileSync(here('build/tundra.rgb')));

// Node 22 has WebAssembly's standardized exception handling behind a flag until 22.x enabled it
const flag = '--experimental-wasm-exnref';
try {
  await WebAssembly.compile(wasm);
} catch (error) {
  if (process.execArgv.includes(flag)) throw error;
  const rerun = spawnSync(process.execPath, [flag, ...process.argv.slice(1)], { stdio: 'inherit' });
  if (rerun.status !== 0) {
    console.log(`Node ${process.version} lacks WebAssembly exception handling (exnref): Node 22 or later needed`);
  }
  process.exit(rerun.status ?? 1);
}

console.log(`Node ${process.version} (V8 ${process.versions.v8})`);
for (const r of await compare(wasm, rgb, 1280, 958)) {
  console.log(`  ${r.workload.padEnd(26)} median ${r.median.toFixed(0).padStart(5)} ms, best ${r.best.toFixed(0).padStart(5)} ms, picture ${r.hash}`);
}

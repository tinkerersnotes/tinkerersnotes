// Dedicated Web Worker that runs the multithreaded module (tiled_seam_carver_wasm.cpp) for the
// page, which talks to it by messages (ThreadedSeamCarver in seam_carver.template.html).
//
// Why not on the page's main thread: the carver's calling thread takes part in the work and
// waits at two barriers per strip of rows. A browser's main thread may not block, so Emscripten
// makes it spin instead, which burns a core the worker threads need and freezes the page for a
// whole step. A worker may block: here the waiting thread sleeps.
//
// Every request carries an id, answered by { id, result } or { id, error }. Requests run one at a
// time in arrival order, since everything but 'init' is synchronous.
'use strict';

let module = null;
const carvers = new Map();  // id from the page -> SeamCarver, or the Error its construction threw

// C++ exceptions arrive as WebAssembly exceptions; their type and message come from the module
function describe(error) {
  if (module && typeof WebAssembly.Exception === 'function' && error instanceof WebAssembly.Exception) {
    const [type, message] = module.getExceptionMessage(error);
    return `${type}: ${message}`;
  }
  return error instanceof Error ? error.message : String(error);
}

function carver(id) {
  const entry = carvers.get(id);
  if (entry instanceof Error) {
    throw entry;
  }
  if (!entry) {
    throw new Error(`no seam carver ${id}`);
  }
  return entry;
}

self.onmessage = async ({ data }) => {
  const { id } = data;
  try {
    let result = null;
    let transfer = [];
    switch (data.op) {
      case 'init':
        // The module's own pthread workers start from the same script: data.moduleUrl is a Blob
        // URL of the module's text, which a worker can import and hand on
        importScripts(data.moduleUrl);
        module = await createTiledSeamCarverModule({ mainScriptUrlOrBlob: data.moduleUrl, poolSize: data.maxThreads - 1 });
        break;
      case 'create':
        try {
          carvers.set(data.carver, new module.SeamCarver(data.rgb, data.width, data.height, data.threads, data.spins));
        } catch (error) {
          carvers.set(data.carver, new Error(describe(error)));  // reported by the carver's next request
          throw error;
        }
        break;
      case 'shrink':
        carver(data.carver).shrink_horizontally(data.pixels, false);
        break;
      case 'rgb':
        result = carver(data.carver).shrunk_rgb();
        transfer = [result.buffer];
        break;
      case 'delete': {
        const entry = carvers.get(data.carver);
        if (entry && !(entry instanceof Error)) {
          entry.delete();
        }
        carvers.delete(data.carver);
        break;
      }
      default:
        throw new Error(`unknown request ${data.op}`);
    }
    self.postMessage({ id, result }, transfer);
  } catch (error) {
    self.postMessage({ id, error: describe(error) });
  }
};

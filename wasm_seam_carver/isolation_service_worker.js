// Service worker that makes the seam carver page cross-origin isolated on hosts that can't send
// the headers for it, such as GitHub Pages. Isolation enables SharedArrayBuffer, which the
// multithreaded WebAssembly module needs for its shared memory.
//
// The page registers this worker and reloads once. From then on the worker answers the page's
// navigations itself, with the network response plus the two headers a server would send:
//   Cross-Origin-Opener-Policy: same-origin     no window of another origin shares the process
//   Cross-Origin-Embedder-Policy: require-corp  every subresource opted in to being embedded
// The page is self-contained, so require-corp blocks nothing it uses. A page opened from disk
// can't have a service worker and stays without isolation (and without threads).
'use strict';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  // Only the page itself needs the headers: other requests go to the network as usual
  if (event.request.mode !== 'navigate') {
    return;
  }
  event.respondWith((async () => {
    const response = await fetch(event.request);
    if (response.type === 'opaqueredirect' || response.status === 0) {
      return response;  // nothing to add headers to
    }
    const headers = new Headers(response.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  })());
});

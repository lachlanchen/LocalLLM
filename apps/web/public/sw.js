const LOCALLLM_CACHE_PREFIX = 'localllm-web-'

self.addEventListener('install', () => {
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) =>
        Promise.all(
          names
            .filter((name) => name.startsWith(LOCALLLM_CACHE_PREFIX))
            .map((name) => caches.delete(name)),
        ),
      )
      .then(() => self.clients.claim()),
  )
})

// Keep application, API, and conversation responses on the normal HTTP path.
// The server already serves fresh HTML and content-addressed immutable assets.
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return

  const url = new URL(event.request.url)
  if (url.origin !== self.location.origin) return

  event.respondWith(fetch(event.request))
})

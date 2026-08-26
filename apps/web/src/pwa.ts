type ServiceWorkerRegistrationLike = {
  update: () => Promise<unknown>
}

type ServiceWorkerRegistrar = {
  register: (
    scriptURL: string,
    options: { scope: string; updateViaCache: 'none' },
  ) => Promise<ServiceWorkerRegistrationLike>
}

type LoadEventSource = {
  addEventListener: (
    type: 'load',
    listener: () => void,
    options: { once: true },
  ) => void
}

export async function registerPwaServiceWorker(
  serviceWorkers: ServiceWorkerRegistrar,
): Promise<void> {
  const registration = await serviceWorkers.register('/sw.js', {
    scope: '/',
    updateViaCache: 'none',
  })
  await registration.update()
}

export function schedulePwaServiceWorkerRegistration(
  loadEvents: LoadEventSource,
  serviceWorkers: ServiceWorkerRegistrar | undefined,
): void {
  if (!serviceWorkers) return

  loadEvents.addEventListener(
    'load',
    () => {
      void registerPwaServiceWorker(serviceWorkers).catch(() => undefined)
    },
    { once: true },
  )
}

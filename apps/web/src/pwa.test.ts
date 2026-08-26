import { describe, expect, it, vi } from 'vitest'
import {
  registerPwaServiceWorker,
  schedulePwaServiceWorkerRegistration,
} from './pwa'

describe('PWA service worker registration', () => {
  it('bypasses the HTTP cache when checking the root service worker', async () => {
    const update = vi.fn().mockResolvedValue(undefined)
    const register = vi.fn().mockResolvedValue({ update })

    await registerPwaServiceWorker({ register })

    expect(register).toHaveBeenCalledWith('/sw.js', {
      scope: '/',
      updateViaCache: 'none',
    })
    expect(update).toHaveBeenCalledOnce()
  })

  it('registers once after load and remains optional on unsupported browsers', async () => {
    let onLoad: (() => void) | undefined
    const addEventListener = vi.fn(
      (_type: 'load', listener: () => void, _options: { once: true }) => {
        onLoad = listener
      },
    )
    const update = vi.fn().mockResolvedValue(undefined)
    const register = vi.fn().mockResolvedValue({ update })

    schedulePwaServiceWorkerRegistration({ addEventListener }, { register })
    expect(register).not.toHaveBeenCalled()
    expect(addEventListener).toHaveBeenCalledWith(
      'load',
      expect.any(Function),
      { once: true },
    )

    onLoad?.()
    await vi.waitFor(() => expect(update).toHaveBeenCalledOnce())

    addEventListener.mockClear()
    schedulePwaServiceWorkerRegistration({ addEventListener }, undefined)
    expect(addEventListener).not.toHaveBeenCalled()
  })
})

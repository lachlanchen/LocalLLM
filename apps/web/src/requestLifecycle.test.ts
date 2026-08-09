import { describe, expect, it } from 'vitest'
import {
  beginRequest,
  createRequestLifecycle,
  finishRequest,
  invalidateRequest,
  isCurrentRequest,
} from './requestLifecycle'

describe('stream request lifecycle', () => {
  it('blocks same-tick duplicate chat or vision runs', () => {
    const lifecycle = createRequestLifecycle()

    const accepted = beginRequest(lifecycle)
    const duplicate = beginRequest(lifecycle)

    expect(accepted).toBe(1)
    expect(duplicate).toBeNull()
    expect(lifecycle.inFlight).toBe(true)
  })

  it('suppresses callbacks from an invalidated stream', () => {
    const lifecycle = createRequestLifecycle()
    const staleGeneration = beginRequest(lifecycle)!

    const currentGeneration = invalidateRequest(lifecycle)

    expect(isCurrentRequest(lifecycle, staleGeneration)).toBe(false)
    expect(finishRequest(lifecycle, staleGeneration)).toBe(false)
    expect(isCurrentRequest(lifecycle, currentGeneration)).toBe(true)
    expect(lifecycle.inFlight).toBe(false)
  })

  it('releases only the matching active stream', () => {
    const lifecycle = createRequestLifecycle()
    const generation = beginRequest(lifecycle)!

    expect(finishRequest(lifecycle, generation + 1)).toBe(false)
    expect(lifecycle.inFlight).toBe(true)
    expect(finishRequest(lifecycle, generation)).toBe(true)
    expect(lifecycle.inFlight).toBe(false)
  })
})

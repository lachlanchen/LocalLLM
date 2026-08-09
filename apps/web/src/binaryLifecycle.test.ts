import { describe, expect, it } from 'vitest'
import {
  abortBinaryOperation,
  beginBinaryOperation,
  createBinaryOperationLifecycle,
  finishBinaryOperation,
  invalidateBinaryOperation,
  isCurrentBinaryOperation,
} from './binaryLifecycle'

describe('binary workbench lifecycle', () => {
  it('admits one upload and blocks rapid replacement or overlapping triage', () => {
    const lifecycle = createBinaryOperationLifecycle()

    const upload = beginBinaryOperation(lifecycle, 'upload')

    expect(upload).toBe(1)
    expect(beginBinaryOperation(lifecycle, 'upload')).toBeNull()
    expect(beginBinaryOperation(lifecycle, 'triage')).toBeNull()
    expect(isCurrentBinaryOperation(lifecycle, upload!, 'upload')).toBe(true)
  })

  it('marks an invalidated upload result for disposal instead of adoption', () => {
    const lifecycle = createBinaryOperationLifecycle()
    const staleUpload = beginBinaryOperation(lifecycle, 'upload')!

    invalidateBinaryOperation(lifecycle)

    expect(isCurrentBinaryOperation(lifecycle, staleUpload, 'upload')).toBe(false)
    expect(finishBinaryOperation(lifecycle, staleUpload)).toBe(false)
    expect(lifecycle.inFlight).toBe(false)
    expect(lifecycle.operation).toBeNull()
  })

  it('releases only the matching operation generation', () => {
    const lifecycle = createBinaryOperationLifecycle()
    const triage = beginBinaryOperation(lifecycle, 'triage')!

    expect(finishBinaryOperation(lifecycle, triage + 1)).toBe(false)
    expect(lifecycle.operation).toBe('triage')
    expect(finishBinaryOperation(lifecycle, triage)).toBe(true)
    expect(lifecycle.operation).toBeNull()
  })

  it('invalidates stale callbacks and aborts transport during unmount cleanup', () => {
    const lifecycle = createBinaryOperationLifecycle()
    const upload = beginBinaryOperation(lifecycle, 'upload')!
    const controller = new AbortController()

    abortBinaryOperation(lifecycle, controller)

    expect(controller.signal.aborted).toBe(true)
    expect(isCurrentBinaryOperation(lifecycle, upload, 'upload')).toBe(false)
    expect(lifecycle.inFlight).toBe(false)
  })
})

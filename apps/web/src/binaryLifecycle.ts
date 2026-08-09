import {
  beginRequest,
  createRequestLifecycle,
  finishRequest,
  invalidateRequest,
  isCurrentRequest,
  type RequestLifecycle,
} from './requestLifecycle'

export type BinaryOperation = 'upload' | 'triage' | 'delete'

export interface BinaryOperationLifecycle extends RequestLifecycle {
  operation: BinaryOperation | null
}

export function createBinaryOperationLifecycle(): BinaryOperationLifecycle {
  return { ...createRequestLifecycle(), operation: null }
}

export function beginBinaryOperation(
  lifecycle: BinaryOperationLifecycle,
  operation: BinaryOperation,
): number | null {
  const generation = beginRequest(lifecycle)
  if (generation !== null) lifecycle.operation = operation
  return generation
}

export function finishBinaryOperation(
  lifecycle: BinaryOperationLifecycle,
  generation: number,
): boolean {
  const current = finishRequest(lifecycle, generation)
  if (current) lifecycle.operation = null
  return current
}

export function invalidateBinaryOperation(lifecycle: BinaryOperationLifecycle): number {
  lifecycle.operation = null
  return invalidateRequest(lifecycle)
}

export function abortBinaryOperation(
  lifecycle: BinaryOperationLifecycle,
  controller?: AbortController | null,
): number {
  const generation = invalidateBinaryOperation(lifecycle)
  controller?.abort()
  return generation
}

export function isCurrentBinaryOperation(
  lifecycle: BinaryOperationLifecycle,
  generation: number,
  operation: BinaryOperation,
): boolean {
  return isCurrentRequest(lifecycle, generation) && lifecycle.operation === operation
}

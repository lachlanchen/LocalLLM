export interface RequestLifecycle {
  generation: number
  inFlight: boolean
}

export function createRequestLifecycle(): RequestLifecycle {
  return { generation: 0, inFlight: false }
}

export function beginRequest(lifecycle: RequestLifecycle): number | null {
  if (lifecycle.inFlight) return null
  lifecycle.inFlight = true
  lifecycle.generation += 1
  return lifecycle.generation
}

export function finishRequest(lifecycle: RequestLifecycle, generation: number): boolean {
  if (lifecycle.generation !== generation) return false
  lifecycle.inFlight = false
  return true
}

export function invalidateRequest(lifecycle: RequestLifecycle): number {
  lifecycle.generation += 1
  lifecycle.inFlight = false
  return lifecycle.generation
}

export function isCurrentRequest(lifecycle: RequestLifecycle, generation: number): boolean {
  return lifecycle.generation === generation
}

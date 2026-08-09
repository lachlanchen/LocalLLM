import {
  beginRequest,
  createRequestLifecycle,
  finishRequest,
  invalidateRequest,
  isCurrentRequest,
  type RequestLifecycle,
} from './requestLifecycle'

export interface ResearchRunGuard extends RequestLifecycle {
  starting: boolean
}

export function createResearchRunGuard(): ResearchRunGuard {
  return { ...createRequestLifecycle(), starting: false }
}

export function beginResearchStart(guard: ResearchRunGuard): number | null {
  const generation = beginRequest(guard)
  if (generation !== null) guard.starting = true
  return generation
}

export function finishResearchStart(guard: ResearchRunGuard, generation: number): boolean {
  const current = finishRequest(guard, generation)
  if (current) guard.starting = false
  return current
}

export function invalidateResearchRun(guard: ResearchRunGuard): number {
  guard.starting = false
  return invalidateRequest(guard)
}

export const isCurrentResearchRun = isCurrentRequest

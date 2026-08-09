import {
  beginRequest,
  createRequestLifecycle,
  finishRequest,
  invalidateRequest,
  isCurrentRequest,
  type RequestLifecycle,
} from './requestLifecycle'

export type McpRequestLane = RequestLifecycle

export const createMcpRequestLane = createRequestLifecycle
export const beginMcpRequest = beginRequest
export const finishMcpRequest = finishRequest
export const isCurrentMcpRequest = isCurrentRequest

export function abortMcpRequest(
  lane: McpRequestLane,
  controller?: AbortController | null,
): number {
  const generation = invalidateRequest(lane)
  controller?.abort()
  return generation
}

import { describe, expect, it } from 'vitest'
import {
  abortMcpRequest,
  beginMcpRequest,
  createMcpRequestLane,
  finishMcpRequest,
  isCurrentMcpRequest,
} from './mcpLifecycle'

describe('MCP request lanes', () => {
  it('prevents a timer or click from overlapping an active status refresh', () => {
    const refreshLane = createMcpRequestLane()
    const first = beginMcpRequest(refreshLane)

    expect(first).toBe(1)
    expect(beginMcpRequest(refreshLane)).toBeNull()
    expect(finishMcpRequest(refreshLane, first!)).toBe(true)
    expect(beginMcpRequest(refreshLane)).toBe(2)
  })

  it('keeps refresh and investigation independent while guarding each lane', () => {
    const refreshLane = createMcpRequestLane()
    const investigationLane = createMcpRequestLane()

    expect(beginMcpRequest(refreshLane)).toBe(1)
    expect(beginMcpRequest(investigationLane)).toBe(1)
    expect(beginMcpRequest(investigationLane)).toBeNull()
  })

  it('aborts transport and suppresses stale completion during cleanup', () => {
    const lane = createMcpRequestLane()
    const generation = beginMcpRequest(lane)!
    const controller = new AbortController()

    abortMcpRequest(lane, controller)

    expect(controller.signal.aborted).toBe(true)
    expect(isCurrentMcpRequest(lane, generation)).toBe(false)
    expect(lane.inFlight).toBe(false)
  })
})

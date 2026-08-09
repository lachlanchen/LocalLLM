import { describe, expect, it } from 'vitest'
import {
  beginResearchStart,
  createResearchRunGuard,
  finishResearchStart,
  invalidateResearchRun,
  isCurrentResearchRun,
} from './researchFlow'

describe('deep-research request lifecycle', () => {
  it('admits only one start request before React can rerender the button', () => {
    const guard = createResearchRunGuard()

    const first = beginResearchStart(guard)
    const duplicate = beginResearchStart(guard)

    expect(first).toBe(1)
    expect(duplicate).toBeNull()
    expect(guard.starting).toBe(true)
    expect(finishResearchStart(guard, first!)).toBe(true)
    expect(beginResearchStart(guard)).toBe(2)
  })

  it('invalidates late create and poll responses after reset', () => {
    const guard = createResearchRunGuard()
    const originalGeneration = beginResearchStart(guard)!

    const resetGeneration = invalidateResearchRun(guard)

    expect(resetGeneration).toBeGreaterThan(originalGeneration)
    expect(isCurrentResearchRun(guard, originalGeneration)).toBe(false)
    expect(finishResearchStart(guard, originalGeneration)).toBe(false)
    expect(isCurrentResearchRun(guard, resetGeneration)).toBe(true)
    expect(guard.starting).toBe(false)
  })
})

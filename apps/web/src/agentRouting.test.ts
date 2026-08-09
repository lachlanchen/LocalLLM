import { describe, expect, it } from 'vitest'
import {
  AGENT_AUTO_STORAGE_KEY,
  MAX_AGENT_GOAL_CHARS,
  readAgentAutoEnabled,
  shouldAutoRouteAgent,
  writeAgentAutoEnabled,
  type AgentPreferenceStorage,
} from './agentRouting'

class MemoryStorage implements AgentPreferenceStorage {
  values = new Map<string, string>()

  getItem(key: string): string | null {
    return this.values.get(key) ?? null
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value)
  }
}

describe('persisted Agent routing preference', () => {
  it('defaults on when the versioned preference is missing or malformed', () => {
    const storage = new MemoryStorage()
    expect(AGENT_AUTO_STORAGE_KEY).toMatch(/\.v1$/)
    expect(readAgentAutoEnabled(storage)).toBe(true)

    storage.setItem(AGENT_AUTO_STORAGE_KEY, 'FALSE')
    expect(readAgentAutoEnabled(storage)).toBe(true)
    storage.setItem(AGENT_AUTO_STORAGE_KEY, '{"enabled":false}')
    expect(readAgentAutoEnabled(storage)).toBe(true)
  })

  it('remembers exact true and false values without storing any task content', () => {
    const storage = new MemoryStorage()
    expect(writeAgentAutoEnabled(false, storage)).toBe(true)
    expect(readAgentAutoEnabled(storage)).toBe(false)
    expect([...storage.values.entries()]).toEqual([[AGENT_AUTO_STORAGE_KEY, 'false']])

    expect(writeAgentAutoEnabled(true, storage)).toBe(true)
    expect(readAgentAutoEnabled(storage)).toBe(true)
    expect([...storage.values.entries()]).toEqual([[AGENT_AUTO_STORAGE_KEY, 'true']])
  })

  it('fails open without throwing when browser storage cannot be read or written', () => {
    const unavailable: AgentPreferenceStorage = {
      getItem: () => { throw new DOMException('blocked', 'SecurityError') },
      setItem: () => { throw new DOMException('blocked', 'SecurityError') },
    }

    expect(readAgentAutoEnabled(unavailable)).toBe(true)
    expect(writeAgentAutoEnabled(false, unavailable)).toBe(false)
    expect(readAgentAutoEnabled(null)).toBe(true)
    expect(writeAgentAutoEnabled(true, null)).toBe(false)
  })
})

describe('explicit isolated-Python routing intent', () => {
  it.each([
    'Run this Python code',
    'execute this snippet',
    'Use Python to calculate the first 20 primes and print them',
    'Run this Python code to parse the following SQL query and print its tables',
    'Use Python to analyze this JavaScript output and print a summary',
    'please run the code',
    'Can you run this program?',
    '运行这段 Python 代码',
    '请执行下面的脚本',
    '請執行這段 Python 程式',
    '用 Python 计算总和并打印结果',
    '```python\nprint(sum(range(10)))\n```\nRun this.',
  ])('routes an explicit execution request: %s', (goal) => {
    expect(shouldAutoRouteAgent(goal)).toBe(true)
  })

  it.each([
    '',
    'Write Python code that calculates the first 20 primes.',
    'Explain this code to me.',
    'Review this Python snippet for bugs.',
    'Do not run this Python code; only explain it.',
    "Don't execute the script.",
    'How do I run this Python code?',
    'Show me how to execute this snippet locally.',
    'What happens if I run this code?',
    'Should I execute this program?',
    'What does "run code" mean?',
    "Explain the phrase 'run code'.",
    '```python\nmessage = "please run the code"\nprint(message)\n```',
    'Please perform a code review.',
    'Execute a plan for improving this project.',
    'Execute this plan.',
    'Run this code review.',
    'Run this Bash script.',
    'Execute this SQL query.',
    'Please run npm test.',
    'Run this code using Bash; do not use Python.',
    'Running code is dangerous.',
    '不要运行这段代码，只解释。',
    '如何执行这段 Python 代码？',
    '写一段 Python 代码计算总和。',
  ])('keeps non-execution or unsupported intent in ordinary chat: %s', (goal) => {
    expect(shouldAutoRouteAgent(goal)).toBe(false)
  })

  it('still identifies oversized execution intent so the caller can show the planner limit', () => {
    expect(shouldAutoRouteAgent(`Run this Python code ${'x'.repeat(MAX_AGENT_GOAL_CHARS)}`)).toBe(true)
    expect(shouldAutoRouteAgent(`\`\`\`python\n${'x'.repeat(MAX_AGENT_GOAL_CHARS)}\n\`\`\`\nRun this.`)).toBe(true)
  })
})

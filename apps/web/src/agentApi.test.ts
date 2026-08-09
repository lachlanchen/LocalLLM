import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  AgentApiError,
  agentApi,
  boundedAgentOutput,
  MAX_AGENT_OUTPUT_CHARS,
  sha256Hex,
} from './agentApi'

afterEach(() => {
  vi.unstubAllGlobals()
})

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('Agent API client', () => {
  it('loads capabilities with cancellation support', async () => {
    const payload = {
      schema_version: '1',
      default_mode: 'ordinary_chat',
      ordinary_chat_auto_executes_tools: false,
      operator_code_execution_enabled: false,
      capabilities: [],
      sandbox_image: 'fixed',
      sandbox_profile: 'python-v1',
      sandbox_ready: false,
      limits: {},
    }
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload))
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()

    await expect(agentApi.capabilities(controller.signal)).resolves.toEqual(payload)
    expect(fetchMock).toHaveBeenCalledWith('/api/agent/capabilities', {
      signal: controller.signal,
    })
  })

  it('proposes from only the current goal, model, and explicit capabilities', async () => {
    const response = {
      planner: 'deterministic-fallback',
      warning: 'Planner fallback.',
      plan: { schema_version: '1', goal: 'Respond safely', steps: [] },
      steps: [],
      events: [],
      executable: false,
    }
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(response))
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()

    await agentApi.propose(
      'Inspect this task',
      'qwen3:8b-q4_K_M',
      ['respond', 'web_search', 'paper_search'],
      controller.signal,
    )

    expect(fetchMock).toHaveBeenCalledWith('/api/agent/plans/propose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        goal: 'Inspect this task',
        model: 'qwen3:8b-q4_K_M',
        enabled_capabilities: ['respond', 'web_search', 'paper_search'],
      }),
      signal: controller.signal,
    })
    const sent = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)
    expect(Object.keys(sent).sort()).toEqual(['enabled_capabilities', 'goal', 'model'])
  })

  it('binds review to exact code before using the execution endpoint', async () => {
    const confirmation = {
      confirmation_token: 'token'.repeat(9),
      code_sha256: 'c2d6e9060cbe8dee44279258cc8677d7a20ec16eeeccfedd09b840283efd3685',
      expires_at: '2026-08-09T10:00:00Z',
      single_use: true,
    }
    const execution = {
      execution_id: `exec_${'a'.repeat(32)}`,
      events: [],
      result: {
        status: 'succeeded',
        exit_code: 0,
        stdout: '42\n',
        stderr: '',
        output_truncated: false,
        duration_ms: 12,
      },
    }
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(confirmation))
      .mockResolvedValueOnce(jsonResponse(execution))
    vi.stubGlobal('fetch', fetchMock)
    const code = 'print(42)'

    expect(await sha256Hex(code)).toBe(confirmation.code_sha256)
    await agentApi.confirmPython(code)
    await agentApi.executePython(code, 3, confirmation.confirmation_token)

    expect(JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)).toEqual({
      tool: 'python',
      code_sha256: confirmation.code_sha256,
      risk_acknowledgement: 'RUN_IN_ISOLATED_SANDBOX',
    })
    expect(JSON.parse((fetchMock.mock.calls[1][1] as RequestInit).body as string)).toEqual({
      tool: 'python',
      code,
      timeout_seconds: 3,
      confirmed: true,
      confirmation_token: confirmation.confirmation_token,
    })
  })

  it('bounds rendered output and presents bounded API errors', async () => {
    expect(boundedAgentOutput(`safe\u0000${'x'.repeat(MAX_AGENT_OUTPUT_CHARS + 20)}`)).toHaveLength(
      MAX_AGENT_OUTPUT_CHARS,
    )
    expect(boundedAgentOutput('safe\u0000text')).toBe('safe�text')

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'disabled' }, 503)))
    await expect(agentApi.capabilities()).rejects.toEqual(
      new AgentApiError(503, 'disabled'),
    )
  })

  it('rejects malformed successful JSON instead of trusting it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('not json', { status: 200 })))
    await expect(agentApi.capabilities()).rejects.toMatchObject({
      name: 'AgentApiError',
      status: 502,
    })
  })
})

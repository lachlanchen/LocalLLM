import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, pullModel, streamChat } from './api'
import type { McpStatus } from './types'

afterEach(() => {
  vi.unstubAllGlobals()
})

function streamedResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
      controller.close()
    },
  }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

describe('local reverse-engineering API client', () => {
  it('loads bridge status and project binaries', async () => {
    const payload: McpStatus = {
      ok: true,
      server: 'PyGhidraMCP',
      version: '1.0.0',
      tool_count: 20,
      read_only_tools: ['decompile_function', 'list_project_binaries'],
      mutation_tools_blocked: Array.from({ length: 8 }, (_, index) => `mutation_${index}`),
      binaries: [{
        name: 'driver.sys',
        file_path: '/project/driver.sys',
        analysis_complete: true,
        code_indexed: true,
        strings_indexed: true,
      }],
      binding: 'loopback-only',
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.mcpStatus()).resolves.toEqual(payload)
    expect(fetchMock).toHaveBeenCalledWith('/api/re/mcp', undefined)
  })

  it('posts only the selected binary, question, and local model', async () => {
    const response = {
      analysis: 'Observed a bounded parser at FUN_0010.',
      evidence: { binary: 'driver.sys' },
      safety: 'Read-only evidence; mutation tools were blocked.',
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.investigateMcp('driver.sys', 'Where is input length validated?', 'localllm-pocket')).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith('/api/re/mcp/investigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        binary_name: 'driver.sys',
        question: 'Where is input length validated?',
        model: 'localllm-pocket',
      }),
    })
  })

  it('deletes a stored inspection artifact by opaque ID', async () => {
    const response = { deleted: true, id: '0123456789abcdefabcd' }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.deleteInspection(response.id)).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith('/api/re/inspect/0123456789abcdefabcd', { method: 'DELETE' })
  })

  it('cancels a live research run before the UI forgets it', async () => {
    const response = { id: 'research123', status: 'cancelled' }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.cancelResearch('research123')).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith('/api/research/research123', { method: 'DELETE' })
  })

  it('keeps the final unterminated chat event and formats structured errors', async () => {
    const tokenFetch = vi.fn().mockResolvedValue(streamedResponse([
      'data: {"choices":[{"delta":{"content":"final"}}]}',
    ]))
    vi.stubGlobal('fetch', tokenFetch)
    let answer = ''
    await streamChat([], 'localllm-pocket', (token) => { answer += token })
    expect(answer).toBe('final')

    const errorFetch = vi.fn().mockResolvedValue(streamedResponse([
      'data: {"error":{"message":"runtime offline","type":"upstream_error"}}',
    ]))
    vi.stubGlobal('fetch', errorFetch)
    await expect(streamChat([], 'localllm-pocket', () => undefined)).rejects.toThrow('runtime offline')
  })

  it('processes a final model-pull event without a trailing separator', async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamedResponse([
      'data: {"status":"complete","completed":10,"total":10}',
    ]))
    vi.stubGlobal('fetch', fetchMock)
    const updates: Array<[number, string]> = []

    await pullModel('qwen3:4b-q4_K_M', (progress, status) => updates.push([progress, status]))

    expect(updates).toEqual([[100, 'complete']])
  })
})

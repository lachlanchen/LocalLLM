import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, imageFileError, pullModel, streamAgentChat, streamChat } from './api'
import type { ChatMessage, McpStatus } from './types'

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

function neverEndingStreamedResponse(chunks: string[]) {
  const encoder = new TextEncoder()
  const cancel = vi.fn()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
    },
    cancel(reason) {
      cancel(reason)
    },
  })
  return {
    body,
    cancel,
    response: new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
  }
}

describe('local reverse-engineering API client', () => {
  it('sends the current revision when deleting a conversation', async () => {
    const response = { deleted: true as const, id: 'conv_0123456789abcdef0123456789abcdef' }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()

    await expect(api.deleteConversation(response.id, 17, controller.signal)).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith(`/api/conversations/${response.id}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expected_revision: 17 }),
      signal: controller.signal,
    })
  })

  it('rejects unsafe or oversized image attachments before reading them', () => {
    expect(imageFileError(new File(['x'], 'payload.svg', { type: 'image/svg+xml' }))).toContain('PNG')
    expect(imageFileError(new File([], 'empty.png', { type: 'image/png' }))).toContain('empty')
    const oversized = new File([new Uint8Array(8 * 1024 * 1024 + 1)], 'large.png', { type: 'image/png' })
    expect(imageFileError(oversized)).toContain('8 MB')
    expect(imageFileError(new File(['safe'], 'photo.webp', { type: 'image/webp' }))).toBeNull()
    expect(imageFileError(new File(['animated'], 'animation.gif', { type: 'image/gif' }))).toContain('PNG')
  })

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

  it('forwards abort signals through MCP refresh and investigation requests', async () => {
    const status = { ok: true, read_only_tools: [], mutation_tools_blocked: [], binding: 'loopback-only' }
    const investigation = { analysis: 'Evidence', evidence: {}, safety: 'Read only' }
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(status), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(investigation), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()

    await api.mcpStatus(controller.signal)
    await api.investigateMcp('driver.sys', 'Where is input validated?', 'localllm-deep', controller.signal)

    expect(fetchMock.mock.calls[0]).toEqual(['/api/re/mcp', { signal: controller.signal }])
    expect(fetchMock.mock.calls[1]).toEqual(['/api/re/mcp/investigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        binary_name: 'driver.sys',
        question: 'Where is input validated?',
        model: 'localllm-deep',
      }),
      signal: controller.signal,
    }])
  })

  it('deletes a stored inspection artifact by opaque ID', async () => {
    const response = { deleted: true, id: '0123456789abcdefabcd' }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.deleteInspection(response.id)).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith('/api/re/inspect/0123456789abcdefabcd', { method: 'DELETE' })
  })

  it('forwards abort signals through binary upload, triage, and deletion', async () => {
    const metadata = {
      id: 'abcdef0123456789abcd',
      filename: 'driver.sys',
      size: 8,
      sha256: 'f'.repeat(64),
      file_type: 'PE binary',
      strings: [],
      strings_truncated: false,
      safety: 'Static inspection only.',
    }
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(metadata), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ analysis: 'bounded parser' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ deleted: true, id: metadata.id }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()
    const file = new File(['MZpayload'], 'driver.sys', { type: 'application/octet-stream' })

    await api.inspectBinary(file, controller.signal)
    await api.triageBinary(metadata, 'localllm-deep', controller.signal)
    await api.deleteInspection(metadata.id, controller.signal)

    const uploadInit = fetchMock.mock.calls[0][1] as RequestInit
    expect(fetchMock.mock.calls[0][0]).toBe('/api/re/inspect')
    expect(uploadInit.method).toBe('POST')
    expect(uploadInit.signal).toBe(controller.signal)
    expect(uploadInit.body).toBeInstanceOf(FormData)
    expect((uploadInit.body as FormData).get('binary')).toBe(file)
    expect(fetchMock.mock.calls[1]).toEqual(['/api/re/triage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata, model: 'localllm-deep' }),
      signal: controller.signal,
    }])
    expect(fetchMock.mock.calls[2]).toEqual([`/api/re/inspect/${metadata.id}`, {
      method: 'DELETE',
      signal: controller.signal,
    }])
  })

  it('cancels a live research run before the UI forgets it', async () => {
    const response = { id: 'research123', status: 'cancelled' }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.cancelResearch('research123')).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith('/api/research/research123', { method: 'DELETE' })
  })

  it('passes cancellation through each research poll and encodes the task ID', async () => {
    const response = { id: 'research/task', status: 'running' }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()

    await expect(api.research('research/task', controller.signal)).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith('/api/research/research%2Ftask', { signal: controller.signal })
  })

  it('requests normalized multi-provider paper evidence independently of the model', async () => {
    const response = {
      query: 'mixture of experts robotics',
      mode: 'papers',
      sources: [{
        title: 'A paper',
        url: 'https://doi.org/10.1/test',
        snippet: 'Evidence',
        provider: 'Crossref',
        providers: ['Crossref'],
        kind: 'paper',
        authors: ['A. Author'],
        year: 2026,
        published_date: '2026-01-01',
        doi: '10.1/test',
        citation_count: 3,
        score: 0.9,
        query: 'mixture of experts robotics',
        provenance: 'crossref:10.1/test',
      }],
      providers: [],
      warnings: [],
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(api.search('mixture of experts robotics', 'papers', 18)).resolves.toEqual(response)
    expect(fetchMock).toHaveBeenCalledWith('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'mixture of experts robotics', mode: 'papers', limit: 18 }),
      signal: undefined,
    })
  })

  it('passes exact research mode and depth values to a persistent run', async () => {
    const response = { id: 'research123', status: 'queued' }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await api.createResearch('Compare current evidence carefully', 'localllm-deep', 'both', 'deep')
    expect(fetchMock).toHaveBeenCalledWith('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: 'Compare current evidence carefully',
        model: 'localllm-deep',
        mode: 'both',
        depth: 'deep',
      }),
    })
  })

  it('accepts a final unterminated DONE event and formats structured errors', async () => {
    const tokenFetch = vi.fn().mockResolvedValue(streamedResponse([
      'data: {"choices":[{"delta":{"content":"final"}}]}\n\ndata: [DONE]',
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

  it('rejects legacy chat EOF without an explicit DONE event', async () => {
    const response = streamedResponse([
      'data: {"choices":[{"delta":{"content":"partial"}}]}',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response))
    let answer = ''

    await expect(streamChat([], 'localllm-fast', (token) => { answer += token }))
      .rejects.toThrow('before a [DONE] event')
    expect(answer).toBe('partial')
    expect(response.body?.locked).toBe(false)
  })

  it('streams agent progress, normalized sources, answer tokens, and vision-safe messages', async () => {
    const response = streamedResponse([
      'event: status\ndata: {"stage":"searching","message":"Searching both"}\n\n',
      'event: source\ndata: {"title":"Paper","url":"https://example.com/paper","snippet":"Evidence","provider":"Crossref","providers":["Crossref"],"kind":"paper","authors":[],"year":2026,"published_date":null,"doi":null,"citation_count":null,"score":1,"query":"q","provenance":[]}\n\n',
      'event: delta\ndata: {"content":"grounded answer"}\n\n',
      'event: done\ndata: {"model":"qwen3-vl:8b","requested_model":"qwen3:8b","mode":"all","sources":[],"providers":[],"warnings":[]}',
    ])
    const fetchMock = vi.fn().mockResolvedValue(response)
    vi.stubGlobal('fetch', fetchMock)
    const statuses: string[] = []
    const sources: string[] = []
    let answer = ''
    const messages: ChatMessage[] = [{
      id: 'image-turn',
      role: 'user',
      content: 'Inspect and research this',
      image: 'data:image/png;base64,AAAA',
    }]

    await streamAgentChat(messages, 'localllm-fast', 'all', {
      onStatus: (event) => statuses.push(event.stage),
      onSource: (source) => sources.push(source.title),
      onToken: (token) => { answer += token },
    })

    expect(statuses).toEqual(['searching'])
    expect(sources).toEqual(['Paper'])
    expect(answer).toBe('grounded answer')
    expect(fetchMock).toHaveBeenCalledWith('/api/agent/chat', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        messages: [{ role: 'user', content: [
          { type: 'text', text: 'Inspect and research this' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } },
        ] }],
        model: 'localllm-fast',
        mode: 'all',
        limit: 18,
        temperature: 0.35,
      }),
    }))
  })

  it('surfaces a typed search clarification while preserving its visible fallback delta', async () => {
    const response = streamedResponse([
      'event: status\ndata: {"stage":"clarifying","message":"Subject needed"}\n\n',
      'event: clarification\ndata: {"reason":"unresolved_search_reference","message":"Which project do you mean?","resolved_mode":"web"}\n\n',
      'event: delta\ndata: {"content":"Which project do you mean?"}\n\n',
      'event: done\ndata: {"model":"qwen3:8b","requested_model":"localllm-fast","mode":"auto","resolved_mode":"web","sources":[],"providers":[],"warnings":[],"clarification":{"reason":"unresolved_search_reference","message":"Which project do you mean?","resolved_mode":"web"}}\n\n',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response))
    const stages: string[] = []
    const clarifications: string[] = []
    let answer = ''

    await streamAgentChat(
      [{ id: 'followup', role: 'user', content: 'What about its latest release?' }],
      'localllm-fast',
      'auto',
      {
        onStatus: (event) => stages.push(event.stage),
        onClarification: (event) => clarifications.push(event.reason),
        onToken: (token) => { answer += token },
      },
    )

    expect(stages).toEqual(['clarifying'])
    expect(clarifications).toEqual(['unresolved_search_reference'])
    expect(answer).toBe('Which project do you mean?')
  })

  it('routes a local vision turn through the guarded agent endpoint', async () => {
    const response = streamedResponse([
      'event: delta\ndata: {"content":"image result"}\n\n',
      'event: done\ndata: {"model":"qwen3-vl:8b","requested_model":"localllm-vision","mode":"local","sources":[],"providers":[],"warnings":[]}\n\n',
    ])
    const fetchMock = vi.fn().mockResolvedValue(response)
    vi.stubGlobal('fetch', fetchMock)
    const message: ChatMessage = {
      id: 'vision-turn',
      role: 'user',
      content: 'Read this diagram',
      image: 'data:image/webp;base64,AAAA',
    }
    let answer = ''
    const controller = new AbortController()

    await streamAgentChat([message], 'localllm-vision', 'local', {
      onToken: (token) => { answer += token },
    }, controller.signal)

    expect(answer).toBe('image result')
    expect(fetchMock).toHaveBeenCalledWith('/api/agent/chat', expect.objectContaining({
      method: 'POST',
      signal: controller.signal,
      body: JSON.stringify({
        messages: [{ role: 'user', content: [
          { type: 'text', text: 'Read this diagram' },
          { type: 'image_url', image_url: { url: 'data:image/webp;base64,AAAA' } },
        ] }],
        model: 'localllm-vision',
        mode: 'local',
        limit: 12,
        temperature: 0.35,
      }),
    }))
  })

  it('fails closed when the agent stream ends without a done event', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamedResponse([
      'event: delta\ndata: {"content":"partial"}\n\n',
    ])))
    await expect(streamAgentChat([], 'localllm-pocket', 'local', { onToken: () => undefined }))
      .rejects.toThrow('ended before a completion event')
  })

  it('cancels and unlocks a never-ending agent stream after its terminal event', async () => {
    const stream = neverEndingStreamedResponse([
      'event: delta\ndata: {"content":"finished"}\n\n',
      'event: done\ndata: {"model":"qwen3:8b","requested_model":"localllm-fast","mode":"local","sources":[],"providers":[],"warnings":[]}\n\n',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(stream.response))
    let answer = ''

    await streamAgentChat([], 'localllm-fast', 'local', {
      onToken: (token) => { answer += token },
    })

    expect(answer).toBe('finished')
    expect(stream.cancel).toHaveBeenCalledTimes(1)
    expect(stream.body.locked).toBe(false)
  })

  it('cancels and unlocks a never-ending agent stream after an error event', async () => {
    const stream = neverEndingStreamedResponse([
      'event: error\ndata: {"message":"agent failed safely"}\n\n',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(stream.response))

    await expect(streamAgentChat([], 'localllm-fast', 'local', { onToken: () => undefined }))
      .rejects.toThrow('agent failed safely')
    expect(stream.cancel).toHaveBeenCalledTimes(1)
    expect(stream.body.locked).toBe(false)
  })

  it('cancels and unlocks agent and legacy chat streams when token handlers throw', async () => {
    const agentStream = neverEndingStreamedResponse([
      'event: delta\ndata: {"content":"token"}\n\n',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(agentStream.response))
    await expect(streamAgentChat([], 'localllm-fast', 'local', {
      onToken: () => { throw new Error('agent renderer failed') },
    })).rejects.toThrow('agent renderer failed')
    expect(agentStream.cancel).toHaveBeenCalledTimes(1)
    expect(agentStream.body.locked).toBe(false)

    const legacyStream = neverEndingStreamedResponse([
      'data: {"choices":[{"delta":{"content":"token"}}]}\n\n',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(legacyStream.response))
    await expect(streamChat([], 'localllm-fast', () => { throw new Error('legacy renderer failed') }))
      .rejects.toThrow('legacy renderer failed')
    expect(legacyStream.cancel).toHaveBeenCalledTimes(1)
    expect(legacyStream.body.locked).toBe(false)
  })

  it('cancels and unlocks a never-ending legacy chat stream at DONE', async () => {
    const stream = neverEndingStreamedResponse(['data: [DONE]\n\n'])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(stream.response))

    await streamChat([], 'localllm-fast', () => undefined)

    expect(stream.cancel).toHaveBeenCalledTimes(1)
    expect(stream.body.locked).toBe(false)
  })

  it('cancels and unlocks model-pull streams after completion or callback failure', async () => {
    const completeStream = neverEndingStreamedResponse([
      'data: {"status":"complete","completed":10,"total":10}\n\n',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(completeStream.response))
    const updates: Array<[number, string]> = []
    await pullModel('qwen3:4b-q4_K_M', (progress, status) => updates.push([progress, status]))
    expect(updates).toEqual([[100, 'complete']])
    expect(completeStream.cancel).toHaveBeenCalledTimes(1)
    expect(completeStream.body.locked).toBe(false)

    const failingStream = neverEndingStreamedResponse([
      'data: {"status":"pulling","completed":5,"total":10}\n\n',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(failingStream.response))
    await expect(pullModel('qwen3:4b-q4_K_M', () => { throw new Error('progress renderer failed') }))
      .rejects.toThrow('progress renderer failed')
    expect(failingStream.cancel).toHaveBeenCalledTimes(1)
    expect(failingStream.body.locked).toBe(false)
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

  it('rejects model-pull EOF without an explicit terminal status', async () => {
    const response = streamedResponse([
      'data: {"status":"pulling","completed":5,"total":10}',
    ])
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response))
    const updates: Array<[number, string]> = []

    await expect(pullModel('qwen3:4b-q4_K_M', (progress, status) => updates.push([progress, status])))
      .rejects.toThrow('before a complete or success event')
    expect(updates).toEqual([[50, 'pulling']])
    expect(response.body?.locked).toBe(false)
  })
})

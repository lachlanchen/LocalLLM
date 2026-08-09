import type {
  AgentClarificationEvent,
  AgentDoneEvent,
  AgentStatusEvent,
  BinaryMetadata,
  CatalogResponse,
  ChatMessage,
  ConversationCompactResponse,
  ConversationFull,
  ConversationListResponse,
  ConversationMessage,
  DeleteConversationResponse,
  DeleteInspectionResponse,
  McpInvestigationResult,
  McpStatus,
  ResearchDepth,
  ResearchTask,
  ResearchSource,
  SearchMode,
  SearchResponse,
  SearchStatus,
  SystemStatus,
} from './types'

const API_BASE = import.meta.env.VITE_API_URL ?? ''
export const MAX_IMAGE_UPLOAD_BYTES = 8 * 1024 * 1024
const SUPPORTED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp'])

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function apiErrorMessage(error: unknown): string {
  if (typeof error === 'string') return error
  if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string') return error.message
  try { return JSON.stringify(error) }
  catch { return String(error) }
}

async function cleanupStreamReader<T>(reader: ReadableStreamDefaultReader<T>, cancel: boolean): Promise<void> {
  try {
    if (cancel) await reader.cancel()
  } catch {
    // Preserve the original parser, callback, or abort error.
  } finally {
    try { reader.releaseLock() }
    catch { /* The stream already released or invalidated the lock. */ }
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const detail = await response.text()
    throw new ApiError(response.status, detail || `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  system: () => request<SystemStatus>('/api/system/status'),
  catalog: () => request<CatalogResponse>('/api/models/catalog'),
  toolchain: () => request<Record<string, Record<string, unknown>>>('/api/re/toolchain'),
  mcpStatus: (signal?: AbortSignal) => request<McpStatus>(
    '/api/re/mcp',
    signal ? { signal } : undefined,
  ),
  investigateMcp: (binaryName: string, question: string, model: string, signal?: AbortSignal) =>
    request<McpInvestigationResult>('/api/re/mcp/investigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ binary_name: binaryName, question, model }),
      ...(signal ? { signal } : {}),
    }),
  searchStatus: () => request<SearchStatus>('/api/search/status'),
  conversations: (signal?: AbortSignal) => request<ConversationListResponse>(
    '/api/conversations',
    signal ? { signal } : undefined,
  ),
  createConversation: (
    payload: {
      title?: string
      model?: string
      mode?: import('./types').ChatMode
      messages?: ConversationMessage[]
    },
    signal?: AbortSignal,
  ) => request<ConversationFull>('/api/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    ...(signal ? { signal } : {}),
  }),
  conversation: (id: string, signal?: AbortSignal) => request<ConversationFull>(
    `/api/conversations/${encodeURIComponent(id)}`,
    signal ? { signal } : undefined,
  ),
  updateConversation: (
    id: string,
    payload: Partial<Pick<ConversationFull, 'title' | 'model' | 'mode' | 'messages'>> & {
      expected_revision: number
    },
    signal?: AbortSignal,
  ) => request<ConversationFull>(`/api/conversations/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    ...(signal ? { signal } : {}),
  }),
  deleteConversation: (id: string, expectedRevision: number, signal?: AbortSignal) =>
    request<DeleteConversationResponse>(`/api/conversations/${encodeURIComponent(id)}`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expected_revision: expectedRevision }),
      ...(signal ? { signal } : {}),
    }),
  compactConversation: (
    id: string,
    model: string,
    keepRecent = 12,
    signal?: AbortSignal,
  ) => request<ConversationCompactResponse>(`/api/conversations/${encodeURIComponent(id)}/compact`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, keep_recent: keepRecent }),
    ...(signal ? { signal } : {}),
  }),
  search: (query: string, mode: SearchMode, limit = 12, signal?: AbortSignal) =>
    request<SearchResponse>('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode, limit }),
      signal,
    }),
  createResearch: (
    question: string,
    model: string,
    mode: SearchMode = 'both',
    depth: ResearchDepth = 'standard',
  ) =>
    request<ResearchTask>('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model, mode, depth }),
    }),
  research: (id: string, signal?: AbortSignal) =>
    request<ResearchTask>(`/api/research/${encodeURIComponent(id)}`, { signal }),
  cancelResearch: (id: string) =>
    request<ResearchTask>(`/api/research/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  inspectBinary: async (file: File, signal?: AbortSignal) => {
    const form = new FormData()
    form.append('binary', file)
    return request<BinaryMetadata>('/api/re/inspect', {
      method: 'POST',
      body: form,
      ...(signal ? { signal } : {}),
    })
  },
  deleteInspection: (id: string, signal?: AbortSignal) =>
    request<DeleteInspectionResponse>(`/api/re/inspect/${encodeURIComponent(id)}`, {
      method: 'DELETE',
      ...(signal ? { signal } : {}),
    }),
  triageBinary: (metadata: BinaryMetadata, model: string, signal?: AbortSignal) =>
    request<{ analysis: string }>('/api/re/triage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata, model }),
      ...(signal ? { signal } : {}),
    }),
}

export async function streamChat(
  messages: ChatMessage[],
  model: string,
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const openAiMessages = toOpenAiMessages(messages)
  const response = await fetch(`${API_BASE}/api/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages: openAiMessages, stream: true, temperature: 0.55 }),
    signal,
  })
  if (!response.ok || !response.body) throw new Error(await response.text())

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed = false
  let endedNaturally = false
  const processEvent = (event: string): boolean => {
    for (const line of event.split('\n')) {
      if (!line.startsWith('data:')) continue
      const data = line.slice(5).trim()
      if (!data) continue
      if (data === '[DONE]') {
        completed = true
        return true
      }
      try {
        const parsed = JSON.parse(data)
        if (parsed.error) throw new Error(apiErrorMessage(parsed.error))
        const token = parsed.choices?.[0]?.delta?.content
        if (typeof token === 'string') onToken(token)
      } catch (error) {
        if (error instanceof SyntaxError) continue
        throw error
      }
    }
    return false
  }
  try {
    let terminal = false
    while (!terminal) {
      const { value, done } = await reader.read()
      if (done) {
        endedNaturally = true
        buffer += decoder.decode()
        if (buffer.trim()) processEvent(buffer)
        break
      }
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''
      for (const event of events) {
        if (processEvent(event)) {
          terminal = true
          break
        }
      }
    }
    if (!completed && !signal?.aborted) throw new Error('The legacy chat stream ended before a [DONE] event.')
  } finally {
    await cleanupStreamReader(reader, !endedNaturally)
  }
}

function toOpenAiMessages(messages: ChatMessage[]) {
  return messages
    .filter((message) => !message.pending)
    .map((message) => ({
      role: message.role,
      content: message.image
        ? [
            { type: 'text', text: message.content },
            { type: 'image_url', image_url: { url: message.image } },
          ]
        : message.content,
    }))
}

export interface AgentChatHandlers {
  onStatus?: (event: AgentStatusEvent) => void
  onClarification?: (event: AgentClarificationEvent) => void
  onSource?: (source: ResearchSource) => void
  onWarning?: (message: string) => void
  onToken: (token: string) => void
  onReasoning?: (token: string) => void
  onDone?: (event: AgentDoneEvent) => void
}

export async function streamAgentChat(
  messages: ChatMessage[],
  model: string,
  mode: import('./types').ChatMode,
  handlers: AgentChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: toOpenAiMessages(messages),
      model,
      mode,
      limit: mode === 'all' || mode === 'auto' ? 18 : 12,
      temperature: 0.35,
    }),
    signal,
  })
  if (!response.ok || !response.body) throw new Error(await response.text())

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed = false
  let endedNaturally = false
  const processEvent = (rawEvent: string): boolean => {
    let eventName = 'message'
    const data: string[] = []
    for (const line of rawEvent.split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim()
      if (line.startsWith('data:')) data.push(line.slice(5).trim())
    }
    if (!data.length) return false
    let payload: Record<string, unknown>
    try { payload = JSON.parse(data.join('\n')) as Record<string, unknown> }
    catch { return false }
    if (eventName === 'error') throw new Error(typeof payload.message === 'string' ? payload.message : 'The local agent stopped unexpectedly.')
    if (eventName === 'status') handlers.onStatus?.(payload as unknown as AgentStatusEvent)
    if (eventName === 'clarification') handlers.onClarification?.(payload as unknown as AgentClarificationEvent)
    if (eventName === 'source') handlers.onSource?.(payload as unknown as ResearchSource)
    if (eventName === 'warning' && typeof payload.message === 'string') handlers.onWarning?.(payload.message)
    if (eventName === 'delta' && typeof payload.content === 'string') handlers.onToken(payload.content)
    if (eventName === 'reasoning' && typeof payload.content === 'string') handlers.onReasoning?.(payload.content)
    if (eventName === 'done') {
      completed = true
      handlers.onDone?.(payload as unknown as AgentDoneEvent)
      return true
    }
    return false
  }
  try {
    let terminal = false
    while (!terminal) {
      const { value, done } = await reader.read()
      if (done) {
        endedNaturally = true
        buffer += decoder.decode()
        if (buffer.trim()) processEvent(buffer)
        break
      }
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''
      for (const event of events) {
        if (processEvent(event)) {
          terminal = true
          break
        }
      }
    }
    if (!completed && !signal?.aborted) throw new Error('The local agent stream ended before a completion event.')
  } finally {
    await cleanupStreamReader(reader, !endedNaturally)
  }
}

export async function pullModel(
  model: string,
  onProgress: (progress: number, status: string) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/models/pull`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  })
  if (!response.ok || !response.body) throw new Error(await response.text())
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed = false
  let endedNaturally = false
  const processEvent = (event: string): boolean => {
    const line = event.split('\n').find((part) => part.startsWith('data:'))
    if (!line) return false
    const payload = JSON.parse(line.slice(5).trim())
    if (payload.error) throw new Error(apiErrorMessage(payload.error))
    const progress = payload.total ? Math.round((payload.completed / payload.total) * 100) : 0
    const terminal = payload.status === 'complete' || payload.status === 'success'
    if (terminal) completed = true
    onProgress(terminal ? 100 : progress, payload.status)
    return terminal
  }
  try {
    let terminal = false
    while (!terminal) {
      const { done, value } = await reader.read()
      if (done) {
        endedNaturally = true
        buffer += decoder.decode()
        if (buffer.trim()) processEvent(buffer)
        break
      }
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''
      for (const event of events) {
        if (processEvent(event)) {
          terminal = true
          break
        }
      }
    }
    if (!completed) throw new Error('The model pull stream ended before a complete or success event.')
  } finally {
    await cleanupStreamReader(reader, !endedNaturally)
  }
}

export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export function imageFileError(file?: File): string | null {
  if (!file) return 'Choose an image to continue.'
  if (!SUPPORTED_IMAGE_TYPES.has(file.type)) return 'Use a PNG, JPEG, or WebP image.'
  if (file.size <= 0) return 'The selected image is empty.'
  if (file.size > MAX_IMAGE_UPLOAD_BYTES) return 'Images must be 8 MB or smaller.'
  return null
}

export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let index = 0
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
  }
  return `${value.toFixed(index > 1 ? 1 : 0)} ${units[index]}`
}

import type {
  BinaryMetadata,
  CatalogResponse,
  ChatMessage,
  DeleteInspectionResponse,
  McpInvestigationResult,
  McpStatus,
  ResearchTask,
  SystemStatus,
} from './types'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

function apiErrorMessage(error: unknown): string {
  if (typeof error === 'string') return error
  if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string') return error.message
  try { return JSON.stringify(error) }
  catch { return String(error) }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  system: () => request<SystemStatus>('/api/system/status'),
  catalog: () => request<CatalogResponse>('/api/models/catalog'),
  toolchain: () => request<Record<string, Record<string, unknown>>>('/api/re/toolchain'),
  mcpStatus: () => request<McpStatus>('/api/re/mcp'),
  investigateMcp: (binaryName: string, question: string, model: string) =>
    request<McpInvestigationResult>('/api/re/mcp/investigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ binary_name: binaryName, question, model }),
    }),
  createResearch: (question: string, model: string) =>
    request<ResearchTask>('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model }),
    }),
  research: (id: string) => request<ResearchTask>(`/api/research/${id}`),
  cancelResearch: (id: string) =>
    request<ResearchTask>(`/api/research/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  inspectBinary: async (file: File) => {
    const form = new FormData()
    form.append('binary', file)
    return request<BinaryMetadata>('/api/re/inspect', { method: 'POST', body: form })
  },
  deleteInspection: (id: string) =>
    request<DeleteInspectionResponse>(`/api/re/inspect/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  triageBinary: (metadata: BinaryMetadata, model: string) =>
    request<{ analysis: string }>('/api/re/triage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata, model }),
    }),
}

export async function streamChat(
  messages: ChatMessage[],
  model: string,
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const openAiMessages = messages
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
  const processEvent = (event: string) => {
    for (const line of event.split('\n')) {
      if (!line.startsWith('data:')) continue
      const data = line.slice(5).trim()
      if (!data || data === '[DONE]') continue
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
  }
  while (true) {
    const { value, done } = await reader.read()
    if (done) {
      buffer += decoder.decode()
      if (buffer.trim()) processEvent(buffer)
      break
    }
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    for (const event of events) processEvent(event)
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
  const processEvent = (event: string) => {
    const line = event.split('\n').find((part) => part.startsWith('data:'))
    if (!line) return
    const payload = JSON.parse(line.slice(5).trim())
    if (payload.error) throw new Error(apiErrorMessage(payload.error))
    const progress = payload.total ? Math.round((payload.completed / payload.total) * 100) : 0
    onProgress(payload.status === 'complete' || payload.status === 'success' ? 100 : progress, payload.status)
  }
  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      buffer += decoder.decode()
      if (buffer.trim()) processEvent(buffer)
      break
    }
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    for (const event of events) processEvent(event)
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

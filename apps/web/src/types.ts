export type ViewId = 'chat' | 'vision' | 'research' | 'models' | 'reverse' | 'api'

export interface ModelInfo {
  id: string
  family: string
  quantization: string
  size_gb: number
  context: number
  modalities: string[]
  tier: string
  role: string
  recommended: boolean
  installed: boolean
}

export interface CatalogResponse {
  models: ModelInfo[]
  installed: Array<{ name?: string; model?: string; size?: number }>
  aliases: Record<string, string>
  planned_download_gb: number
}

export interface SystemStatus {
  service: { ok: boolean; version: string }
  gpu: {
    ok: boolean
    devices: Array<{
      index: number
      name: string
      memory_total_mb: number
      memory_used_mb: number
      memory_free_mb: number
      temperature_c: number
      power_w: number
    }>
    error?: string
    diagnosis?: string
  }
  ollama: { ok: boolean; version?: string; error?: string }
  storage: { total: number; used: number; free: number }
  binding: { host: string; port: number; local_only: boolean }
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  image?: string
  pending?: boolean
}

export interface ResearchSource {
  title: string
  url: string
  snippet: string
}

export interface ResearchTask {
  id: string
  question: string
  model: string
  status: 'queued' | 'running' | 'complete' | 'failed'
  stage: string
  progress: number
  queries: string[]
  sources: ResearchSource[]
  report: string
  error?: string
}

export interface BinaryMetadata {
  id: string
  filename: string
  size: number
  sha256: string
  file_type: string
  strings: string[]
  strings_truncated: boolean
  safety: string
}


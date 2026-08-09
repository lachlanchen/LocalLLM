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
  ollama?: { ok: boolean; error?: string }
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
  model?: string
  mode?: ChatMode
  sources?: ResearchSource[]
  activity?: string[]
  warning?: string
}

export type ChatMode = 'local' | 'web' | 'papers' | 'all'
export type SearchMode = 'web' | 'papers' | 'both'
export type ResearchDepth = 'quick' | 'standard' | 'deep'

export interface ResearchSource {
  title: string
  url: string
  snippet: string
  provider?: string
  providers?: string[]
  kind?: 'web' | 'paper' | string
  authors?: string[]
  year?: number | null
  published_date?: string | null
  doi?: string | null
  citation_count?: number | null
  score?: number | null
  query?: string
  provenance?: SourceProvenance[]
}

export interface SourceProvenance {
  provider: string
  query: string
  record_id: string | null
  retrieved_at: string
}

export interface SearchProviderStatus {
  name: string
  kind: string
  enabled: boolean
  configured: boolean
  requires_key: boolean
  description: string
}

export interface SearchProviderRun {
  name: string
  kind: string
  ok: boolean
  result_count: number
  duration_ms: number
  error?: string
}

export interface SearchStatus {
  providers: SearchProviderStatus[]
  modes: SearchMode[]
  limits: {
    max_results: number
    max_concurrency: number
    provider_timeout_seconds: number
  }
}

export interface SearchResponse {
  query: string
  mode: SearchMode
  sources: ResearchSource[]
  providers: SearchProviderRun[]
  warnings: string[]
}

export interface AgentStatusEvent {
  stage: string
  message: string
  model?: string
}

export interface AgentDoneEvent {
  model: string
  requested_model: string
  mode: ChatMode
  sources: ResearchSource[]
  providers: SearchProviderRun[]
  warnings: string[]
}

export interface ResearchTask {
  id: string
  question: string
  model: string
  status: 'queued' | 'running' | 'complete' | 'failed' | 'cancelled'
  mode: SearchMode
  depth: ResearchDepth
  stage: string
  progress: number
  queries: string[]
  sources: ResearchSource[]
  providers: SearchProviderRun[]
  provider_errors: string[]
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

export interface DeleteInspectionResponse {
  deleted: true
  id: string
}

export interface McpProjectBinary {
  name: string
  file_path: string
  analysis_complete: boolean
  code_indexed: boolean
  strings_indexed: boolean
}

export interface McpStatus {
  ok: boolean
  server?: string
  version?: string
  tool_count?: number
  read_only_tools: string[]
  mutation_tools_blocked: string[] | boolean
  binaries?: McpProjectBinary[]
  binding: string
  error?: string
}

export interface McpInvestigationResult {
  binary?: string
  question?: string
  analysis: string
  evidence: Record<string, unknown>
  safety: string
}

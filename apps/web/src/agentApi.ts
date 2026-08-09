export type AgentCapability = 'respond' | 'web_search' | 'paper_search' | 'vision' | 'python'

export interface AgentCapabilityStatus {
  id: 'plan_validation' | 'python'
  available: boolean
  default_enabled: false
  invocation: 'explicit_endpoint' | 'two_step_confirmation'
  reason?: string | null
}

export interface AgentCapabilities {
  schema_version: '1'
  default_mode: 'ordinary_chat'
  ordinary_chat_auto_executes_tools: false
  operator_code_execution_enabled: boolean
  capabilities: AgentCapabilityStatus[]
  sandbox_image: string
  sandbox_profile: string
  sandbox_ready: boolean
  limits: {
    network: 'none'
    host_mounts: false
    root_filesystem: 'read_only'
    user: '65532:65532'
    capabilities: 'dropped'
    no_new_privileges: true
    workdir: 'ephemeral_tmpfs'
    pids: 64
    memory_mib: 512
    cpus: 1
    max_output_bytes: 65536
    max_seconds: 20
    max_parallel: 2
  }
}

interface AgentStepBase {
  id: string
  objective: string
  depends_on: string[]
}

export interface RespondPlanStep extends AgentStepBase {
  capability: 'respond'
  arguments: Record<string, never>
}

export interface SearchPlanStep extends AgentStepBase {
  capability: 'web_search' | 'paper_search'
  arguments: { query: string; limit: number }
}

export interface VisionPlanStep extends AgentStepBase {
  capability: 'vision'
  arguments: { image_ids: string[]; question: string }
}

export interface PythonPlanStep extends AgentStepBase {
  capability: 'python'
  arguments: { code: string; timeout_seconds: number }
}

export type AgentPlanStep = RespondPlanStep | SearchPlanStep | VisionPlanStep | PythonPlanStep

export interface StagedAgentStep extends AgentStepBase {
  capability: AgentCapability
  state: 'ready' | 'awaiting_explicit_confirmation'
}

export interface AgentPlanProposal {
  planner: 'local-model' | 'deterministic-fallback'
  warning: string | null
  plan: {
    schema_version: '1'
    goal: string
    steps: AgentPlanStep[]
  }
  steps: StagedAgentStep[]
  events: Array<{
    type: 'plan.staged'
    schema_version: '1'
    step_count: number
    capabilities: AgentCapability[]
  }>
  executable: false
}

export interface AgentCodeConfirmation {
  confirmation_token: string
  code_sha256: string
  expires_at: string
  single_use: true
}

export type AgentExecutionStatus =
  | 'succeeded'
  | 'failed'
  | 'timed_out'
  | 'output_limited'
  | 'sandbox_error'

export interface AgentCodeExecution {
  execution_id: string
  events: Array<Record<string, unknown> & { type: string }>
  result: {
    status: AgentExecutionStatus
    exit_code: number | null
    stdout: string
    stderr: string
    output_truncated: boolean
    duration_ms: number
  }
}

const API_BASE = import.meta.env.VITE_API_URL ?? ''
const MAX_ERROR_CHARS = 500
export const MAX_AGENT_CODE_BYTES = 32 * 1024
export const MAX_AGENT_OUTPUT_CHARS = 65_536

export class AgentApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'AgentApiError'
    this.status = status
  }
}

function boundedError(body: string, fallback: string): string {
  if (!body) return fallback
  try {
    const parsed = JSON.parse(body) as { detail?: unknown }
    if (typeof parsed.detail === 'string') return parsed.detail.slice(0, MAX_ERROR_CHARS)
  } catch {
    // The plain response body is handled below.
  }
  return body.slice(0, MAX_ERROR_CHARS)
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  const body = await response.text()
  if (!response.ok) {
    throw new AgentApiError(
      response.status,
      boundedError(body, `${response.status} ${response.statusText}`),
    )
  }
  try {
    return JSON.parse(body) as T
  } catch {
    throw new AgentApiError(502, 'The local Agent API returned malformed JSON.')
  }
}

export function boundedAgentOutput(value: unknown): string {
  if (typeof value !== 'string') return ''
  return value
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\u202A-\u202E\u2066-\u2069]/g, '�')
    .slice(0, MAX_AGENT_OUTPUT_CHARS)
}

export async function sha256Hex(code: string): Promise<string> {
  const encoded = new TextEncoder().encode(code)
  if (!encoded.length || encoded.byteLength > MAX_AGENT_CODE_BYTES) {
    throw new AgentApiError(422, 'Python code exceeds the local sandbox input limit.')
  }
  if (!globalThis.crypto?.subtle) {
    throw new AgentApiError(500, 'This browser cannot bind confirmation to the reviewed code.')
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', encoded)
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

export const agentApi = {
  capabilities: (signal?: AbortSignal) => requestJson<AgentCapabilities>(
    '/api/agent/capabilities',
    signal ? { signal } : undefined,
  ),

  propose: (
    goal: string,
    model: string,
    enabledCapabilities: AgentCapability[],
    signal?: AbortSignal,
  ) => requestJson<AgentPlanProposal>('/api/agent/plans/propose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      goal,
      model,
      enabled_capabilities: enabledCapabilities,
    }),
    ...(signal ? { signal } : {}),
  }),

  confirmPython: async (code: string, signal?: AbortSignal) => {
    const codeSha256 = await sha256Hex(code)
    const confirmation = await requestJson<AgentCodeConfirmation>('/api/agent/code/confirmations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tool: 'python',
        code_sha256: codeSha256,
        risk_acknowledgement: 'RUN_IN_ISOLATED_SANDBOX',
      }),
      ...(signal ? { signal } : {}),
    })
    if (
      confirmation.code_sha256 !== codeSha256
      || typeof confirmation.confirmation_token !== 'string'
      || confirmation.confirmation_token.length < 32
    ) {
      throw new AgentApiError(502, 'The local Agent API returned an invalid code confirmation.')
    }
    return confirmation
  },

  executePython: (
    code: string,
    timeoutSeconds: number,
    confirmationToken: string,
    signal?: AbortSignal,
  ) => requestJson<AgentCodeExecution>('/api/agent/code/executions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tool: 'python',
      code,
      timeout_seconds: timeoutSeconds,
      confirmed: true,
      confirmation_token: confirmationToken,
    }),
    ...(signal ? { signal } : {}),
  }),
}

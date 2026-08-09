const API_BASE = import.meta.env.VITE_API_URL ?? ''
const JOB_ID_PATTERN = /^img_[0-9a-f]{32}$/
const MAX_ERROR_CHARS = 500
const MAX_IMAGE_BYTES = 32 * 1024 * 1024

export type ImageJobState = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
export type ImageOutputFormat = 'png' | 'jpeg'

export interface ImageGenerationStatus {
  enabled: boolean
  available: boolean
  runtime_ready: boolean
  model_ready: boolean
  gpu_ready: boolean
  gpu_capacity_ready: boolean
  gpu_free_memory_bytes: number | null
  minimum_gpu_free_memory_bytes: number
  worker_running: boolean
  model: {
    id: string
    revision: string
    license: string
    parameters: number
  }
  gpu: number
  limits: {
    concurrency: number
    pending_jobs: number
    timeout_seconds: number
    output_quota_bytes: number
    max_output_files: number
    max_output_bytes: number
    idle_unload_seconds: number
  }
  usage: {
    queued: number
    running: number
    output_files: number
    output_bytes: number
  }
}

export interface ImageGenerationJob {
  id: string
  status: ImageJobState
  created_at: number
  started_at: number | null
  completed_at: number | null
  width: number
  height: number
  steps: number
  seed: number
  output_format: ImageOutputFormat
  image_url: string | null
  error: string | null
  duration_ms: number | null
  peak_gpu_memory_bytes: number | null
  settings_known: boolean
}

export interface CreateImageGenerationJob {
  prompt: string
  width: 512 | 768 | 1024
  height: 512 | 768 | 1024
  steps: number
  seed: number
  output_format: ImageOutputFormat
  jpeg_quality: number
}

export class ImageGenerationApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ImageGenerationApiError'
    this.status = status
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new ImageGenerationApiError(502, 'The local image API returned malformed data.')
  }
  return value as Record<string, unknown>
}

function boundedError(body: string, fallback: string): string {
  if (!body) return fallback
  try {
    const payload = asRecord(JSON.parse(body))
    if (typeof payload.detail === 'string') return payload.detail.slice(0, MAX_ERROR_CHARS)
  } catch {
    // The bounded plain response body is used below.
  }
  return body.slice(0, MAX_ERROR_CHARS)
}

async function requestPayload(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(`${API_BASE}${path}`, init)
  const body = await response.text()
  if (!response.ok) {
    throw new ImageGenerationApiError(
      response.status,
      boundedError(body, `${response.status} ${response.statusText}`),
    )
  }
  try {
    return JSON.parse(body) as unknown
  } catch {
    throw new ImageGenerationApiError(502, 'The local image API returned malformed JSON.')
  }
}

function requireBoolean(record: Record<string, unknown>, key: string): boolean {
  const value = record[key]
  if (typeof value !== 'boolean') {
    throw new ImageGenerationApiError(502, 'The local image API returned malformed data.')
  }
  return value
}

function requireNumber(record: Record<string, unknown>, key: string): number {
  const value = record[key]
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ImageGenerationApiError(502, 'The local image API returned malformed data.')
  }
  return value
}

function optionalNumber(record: Record<string, unknown>, key: string): number | null {
  const value = record[key]
  if (value === null) return null
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ImageGenerationApiError(502, 'The local image API returned malformed data.')
  }
  return value
}

function optionalString(record: Record<string, unknown>, key: string): string | null {
  const value = record[key]
  if (value === null) return null
  if (typeof value !== 'string' || value.length > 1000) {
    throw new ImageGenerationApiError(502, 'The local image API returned malformed data.')
  }
  return value
}

function parseJob(value: unknown): ImageGenerationJob {
  const job = asRecord(value)
  const id = job.id
  const status = job.status
  const outputFormat = job.output_format
  if (typeof id !== 'string' || !JOB_ID_PATTERN.test(id)) {
    throw new ImageGenerationApiError(502, 'The local image API returned an invalid job ID.')
  }
  if (!['queued', 'running', 'succeeded', 'failed', 'cancelled'].includes(String(status))) {
    throw new ImageGenerationApiError(502, 'The local image API returned an invalid job state.')
  }
  if (outputFormat !== 'png' && outputFormat !== 'jpeg') {
    throw new ImageGenerationApiError(502, 'The local image API returned an invalid image type.')
  }
  return {
    id,
    status: status as ImageJobState,
    created_at: requireNumber(job, 'created_at'),
    started_at: optionalNumber(job, 'started_at'),
    completed_at: optionalNumber(job, 'completed_at'),
    width: requireNumber(job, 'width'),
    height: requireNumber(job, 'height'),
    steps: requireNumber(job, 'steps'),
    seed: requireNumber(job, 'seed'),
    output_format: outputFormat,
    image_url: optionalString(job, 'image_url'),
    error: optionalString(job, 'error'),
    duration_ms: optionalNumber(job, 'duration_ms'),
    peak_gpu_memory_bytes: optionalNumber(job, 'peak_gpu_memory_bytes'),
    settings_known: requireBoolean(job, 'settings_known'),
  }
}

function parseJobs(value: unknown): ImageGenerationJob[] {
  if (!Array.isArray(value) || value.length > 128) {
    throw new ImageGenerationApiError(502, 'The local image API returned an invalid job list.')
  }
  return value.map(parseJob)
}

function parseStatus(value: unknown): ImageGenerationStatus {
  const status = asRecord(value)
  const model = asRecord(status.model)
  const limits = asRecord(status.limits)
  const usage = asRecord(status.usage)
  for (const key of ['id', 'revision', 'license'] as const) {
    if (typeof model[key] !== 'string' || model[key].length > 200) {
      throw new ImageGenerationApiError(502, 'The local image API returned malformed model data.')
    }
  }
  return {
    enabled: requireBoolean(status, 'enabled'),
    available: requireBoolean(status, 'available'),
    runtime_ready: requireBoolean(status, 'runtime_ready'),
    model_ready: requireBoolean(status, 'model_ready'),
    gpu_ready: requireBoolean(status, 'gpu_ready'),
    gpu_capacity_ready: requireBoolean(status, 'gpu_capacity_ready'),
    gpu_free_memory_bytes: optionalNumber(status, 'gpu_free_memory_bytes'),
    minimum_gpu_free_memory_bytes: requireNumber(status, 'minimum_gpu_free_memory_bytes'),
    worker_running: requireBoolean(status, 'worker_running'),
    model: {
      id: model.id as string,
      revision: model.revision as string,
      license: model.license as string,
      parameters: requireNumber(model, 'parameters'),
    },
    gpu: requireNumber(status, 'gpu'),
    limits: {
      concurrency: requireNumber(limits, 'concurrency'),
      pending_jobs: requireNumber(limits, 'pending_jobs'),
      timeout_seconds: requireNumber(limits, 'timeout_seconds'),
      output_quota_bytes: requireNumber(limits, 'output_quota_bytes'),
      max_output_files: requireNumber(limits, 'max_output_files'),
      max_output_bytes: requireNumber(limits, 'max_output_bytes'),
      idle_unload_seconds: requireNumber(limits, 'idle_unload_seconds'),
    },
    usage: {
      queued: requireNumber(usage, 'queued'),
      running: requireNumber(usage, 'running'),
      output_files: requireNumber(usage, 'output_files'),
      output_bytes: requireNumber(usage, 'output_bytes'),
    },
  }
}

function authorizedHeaders(apiKey: string): HeadersInit {
  return apiKey ? { 'Content-Type': 'application/json', 'X-LocalLLM-Key': apiKey } : {
    'Content-Type': 'application/json',
  }
}

function readHeaders(apiKey: string): HeadersInit | undefined {
  return apiKey ? { 'X-LocalLLM-Key': apiKey } : undefined
}

function validatedJobPath(id: string, suffix = ''): string {
  if (!JOB_ID_PATTERN.test(id)) {
    throw new ImageGenerationApiError(400, 'Invalid local image job ID.')
  }
  return `/api/images/jobs/${id}${suffix}`
}

export const imageGenerationApi = {
  status: async (signal?: AbortSignal) => parseStatus(
    await requestPayload('/api/images/status', signal ? { signal } : undefined),
  ),
  create: async (payload: CreateImageGenerationJob, apiKey: string, signal?: AbortSignal) => parseJob(
    await requestPayload('/api/images/jobs', {
      method: 'POST',
      headers: authorizedHeaders(apiKey),
      body: JSON.stringify(payload),
      ...(signal ? { signal } : {}),
    }),
  ),
  jobs: async (apiKey: string, signal?: AbortSignal) => parseJobs(
    await requestPayload('/api/images/jobs', {
      headers: readHeaders(apiKey),
      ...(signal ? { signal } : {}),
    }),
  ),
  job: async (id: string, apiKey: string, signal?: AbortSignal) => parseJob(
    await requestPayload(validatedJobPath(id), {
      headers: readHeaders(apiKey),
      ...(signal ? { signal } : {}),
    }),
  ),
  delete: async (id: string, apiKey: string, signal?: AbortSignal): Promise<void> => {
    const response = await fetch(`${API_BASE}${validatedJobPath(id)}`, {
      method: 'DELETE',
      headers: apiKey ? { 'X-LocalLLM-Key': apiKey } : undefined,
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw new ImageGenerationApiError(
        response.status,
        boundedError(body, `${response.status} ${response.statusText}`),
      )
    }
  },
  imageBlob: async (id: string, apiKey: string, signal?: AbortSignal): Promise<Blob> => {
    const response = await fetch(`${API_BASE}${validatedJobPath(id, '/image')}`, {
      headers: readHeaders(apiKey),
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) {
      const body = await response.text()
      throw new ImageGenerationApiError(
        response.status,
        boundedError(body, `${response.status} ${response.statusText}`),
      )
    }
    const declared = Number(response.headers.get('content-length') ?? '0')
    if (declared && (!Number.isSafeInteger(declared) || declared < 1 || declared > MAX_IMAGE_BYTES)) {
      throw new ImageGenerationApiError(502, 'The local image output exceeded its size limit.')
    }
    const blob = await response.blob()
    if (!['image/png', 'image/jpeg'].includes(blob.type) || blob.size < 1 || blob.size > MAX_IMAGE_BYTES) {
      throw new ImageGenerationApiError(502, 'The local image output failed type or size validation.')
    }
    return blob
  },
  unload: async (apiKey: string, signal?: AbortSignal): Promise<boolean> => {
    const payload = asRecord(await requestPayload('/api/images/unload', {
      method: 'POST',
      headers: apiKey ? { 'X-LocalLLM-Key': apiKey } : undefined,
      ...(signal ? { signal } : {}),
    }))
    if (typeof payload.released !== 'boolean' || payload.worker_running !== false) {
      throw new ImageGenerationApiError(502, 'The local image API returned malformed unload data.')
    }
    return payload.released
  },
}

export async function unloadAndVerifyImageWorker(
  apiKey: string,
  signal?: AbortSignal,
): Promise<ImageGenerationStatus> {
  await imageGenerationApi.unload(apiKey, signal)
  const status = await imageGenerationApi.status(signal)
  if (status.worker_running) {
    throw new ImageGenerationApiError(502, 'The local image worker did not release its GPU memory.')
  }
  return status
}

export function isTerminalImageJob(status: ImageJobState): boolean {
  return status === 'succeeded' || status === 'failed' || status === 'cancelled'
}

import { Download, ImageIcon, LoaderCircle, RefreshCw, Sparkles, Trash2, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  imageGenerationApi,
  isTerminalImageJob,
  unloadAndVerifyImageWorker,
  type ImageGenerationJob,
  type ImageGenerationStatus,
} from './imageGenerationApi'
import './ImageGenerationPanel.css'

type ImagePreset = 512 | 768 | 1024

export function describeImageGenerationStatus(status: ImageGenerationStatus | null): string {
  if (status === null) return 'Open this panel to check the optional local image worker.'
  if (!status.enabled) return 'Disabled by the operator. Installation alone never enables this lane.'
  if (!status.runtime_ready) return 'The isolated image runtime is not installed.'
  if (!status.model_ready) return 'The pinned Z-Image-Turbo model is not downloaded.'
  if (!status.gpu_ready) return `Physical GPU ${status.gpu} is not available to the image worker.`
  if (!status.gpu_capacity_ready) {
    const free = status.gpu_free_memory_bytes === null
      ? 'an unknown amount of memory'
      : `${(status.gpu_free_memory_bytes / 1024 ** 3).toFixed(1)} GiB free`
    const required = (status.minimum_gpu_free_memory_bytes / 1024 ** 3).toFixed(0)
    return `GPU ${status.gpu} has ${free}; close or move GPU workloads until at least ${required} GiB is free.`
  }
  if (!status.available) return 'The local image lane failed a storage or runtime readiness check.'
  return status.worker_running
    ? `Ready on GPU ${status.gpu}; the model is warm.`
    : `Ready on GPU ${status.gpu}; the first image will load the model.`
}

export function imageGenerationPanelIsBusy(
  status: ImageGenerationStatus | null,
  job: ImageGenerationJob | null,
  mutating: boolean,
): boolean {
  return mutating
    || job?.status === 'queued'
    || job?.status === 'running'
    || status?.worker_running === true
}

function displayError(error: unknown): string {
  if (error instanceof Error && error.name === 'AbortError') return ''
  if (error instanceof Error) return error.message.slice(0, 500)
  return 'The local image request failed.'
}

function waitForPoll(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, 1000)
    signal.addEventListener('abort', () => {
      window.clearTimeout(timer)
      reject(new DOMException('Polling cancelled', 'AbortError'))
    }, { once: true })
  })
}

export function ImageGenerationPanel({
  disabled = false,
  onUseResult,
  onBusyChange,
}: {
  disabled?: boolean
  onUseResult?: (image: Blob, prompt: string) => void | Promise<void>
  onBusyChange?: (busy: boolean) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [status, setStatus] = useState<ImageGenerationStatus | null>(null)
  const [statusBusy, setStatusBusy] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [preset, setPreset] = useState<ImagePreset>(1024)
  const [seed, setSeed] = useState('42')
  const [apiKey, setApiKey] = useState(
    import.meta.env.VITE_LOCALLLM_API_KEY ?? 'local-dev-key',
  )
  const [job, setJob] = useState<ImageGenerationJob | null>(null)
  const [jobs, setJobs] = useState<ImageGenerationJob[]>([])
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewBusy, setPreviewBusy] = useState(false)
  const [mutating, setMutating] = useState(false)
  const [error, setError] = useState('')
  const statusAbort = useRef<AbortController | null>(null)
  const mutationAbort = useRef<AbortController | null>(null)
  const pollAbort = useRef<AbortController | null>(null)
  const previewAbort = useRef<AbortController | null>(null)
  const previewUrlRef = useRef<string | null>(null)
  const inMemoryPromptsRef = useRef(new Map<string, string>())
  const onBusyChangeRef = useRef(onBusyChange)
  const active = job?.status === 'queued' || job?.status === 'running'
  const busy = imageGenerationPanelIsBusy(status, job, mutating)

  const refreshStatus = async () => {
    statusAbort.current?.abort()
    const controller = new AbortController()
    statusAbort.current = controller
    setStatusBusy(true)
    try {
      const nextStatus = await imageGenerationApi.status(controller.signal)
      setStatus(nextStatus)
      const nextJobs = await imageGenerationApi.jobs(apiKey, controller.signal)
      setJobs(nextJobs)
      setJob((current) => {
        if (current) return nextJobs.find((candidate) => candidate.id === current.id) ?? null
        return nextJobs[0] ?? null
      })
      setError('')
    } catch (caught) {
      const message = displayError(caught)
      if (message) setError(message)
    } finally {
      if (statusAbort.current === controller) {
        statusAbort.current = null
        setStatusBusy(false)
      }
    }
  }

  useEffect(() => {
    if (!disabled) void refreshStatus()
    return () => statusAbort.current?.abort()
  }, [disabled])

  useEffect(() => {
    if (disabled || !job || isTerminalImageJob(job.status)) return
    pollAbort.current?.abort()
    const controller = new AbortController()
    pollAbort.current = controller
    void (async () => {
      try {
        let current = job
        while (!isTerminalImageJob(current.status)) {
          await waitForPoll(controller.signal)
          current = await imageGenerationApi.job(current.id, apiKey, controller.signal)
          setJob(current)
          setJobs((previous) => [current, ...previous.filter((item) => item.id !== current.id)])
        }
        await refreshStatus()
      } catch (caught) {
        const message = displayError(caught)
        if (message) setError(message)
      } finally {
        if (pollAbort.current === controller) pollAbort.current = null
      }
    })()
    return () => controller.abort()
  }, [apiKey, disabled, job?.id, job?.status])

  useEffect(() => {
    previewAbort.current?.abort()
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    previewUrlRef.current = null
    setPreviewUrl(null)
    setPreviewBusy(false)
    if (!expanded || disabled || job?.status !== 'succeeded') return
    const controller = new AbortController()
    previewAbort.current = controller
    setPreviewBusy(true)
    void imageGenerationApi.imageBlob(job.id, apiKey, controller.signal)
      .then((blob) => {
        const nextUrl = URL.createObjectURL(blob)
        if (controller.signal.aborted) {
          URL.revokeObjectURL(nextUrl)
          return
        }
        previewUrlRef.current = nextUrl
        setPreviewUrl(nextUrl)
      })
      .catch((caught) => {
        const message = displayError(caught)
        if (message) setError(message)
      })
      .finally(() => {
        if (previewAbort.current === controller) {
          previewAbort.current = null
          setPreviewBusy(false)
        }
      })
    return () => controller.abort()
  }, [apiKey, disabled, expanded, job?.id, job?.status])

  useEffect(() => {
    if (disabled || !status?.worker_running || active) return
    const timer = window.setInterval(() => void refreshStatus(), 5000)
    return () => window.clearInterval(timer)
  }, [active, disabled, status?.worker_running])

  useEffect(() => {
    if (!disabled) return
    statusAbort.current?.abort()
    mutationAbort.current?.abort()
    pollAbort.current?.abort()
    previewAbort.current?.abort()
    setStatusBusy(false)
    setMutating(false)
  }, [disabled])

  useEffect(() => () => {
    statusAbort.current?.abort()
    mutationAbort.current?.abort()
    pollAbort.current?.abort()
    previewAbort.current?.abort()
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
  }, [])

  useEffect(() => {
    onBusyChangeRef.current = onBusyChange
    onBusyChange?.(busy)
  }, [busy, onBusyChange])

  useEffect(() => () => onBusyChangeRef.current?.(false), [])

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (disabled || mutating || !status?.available || !prompt.trim()) return
    mutationAbort.current?.abort()
    pollAbort.current?.abort()
    const controller = new AbortController()
    mutationAbort.current = controller
    setMutating(true)
    setError('')
    try {
      const parsedSeed = Number(seed)
      if (!Number.isSafeInteger(parsedSeed) || parsedSeed < 0 || parsedSeed > 4_294_967_295) {
        throw new Error('Seed must be an integer from 0 through 4,294,967,295.')
      }
      const created = await imageGenerationApi.create({
        prompt: prompt.trim(),
        width: preset,
        height: preset,
        steps: 9,
        seed: parsedSeed,
        output_format: 'png',
        jpeg_quality: 90,
      }, apiKey, controller.signal)
      inMemoryPromptsRef.current.set(created.id, prompt.trim())
      setJob(created)
      setJobs((previous) => [created, ...previous.filter((item) => item.id !== created.id)])
    } catch (caught) {
      const message = displayError(caught)
      if (message) setError(message)
    } finally {
      if (mutationAbort.current === controller) {
        mutationAbort.current = null
        setMutating(false)
      }
    }
  }

  const removeJob = async () => {
    if (disabled || !job || mutating) return
    mutationAbort.current?.abort()
    pollAbort.current?.abort()
    const controller = new AbortController()
    mutationAbort.current = controller
    setMutating(true)
    setError('')
    try {
      await imageGenerationApi.delete(job.id, apiKey, controller.signal)
      inMemoryPromptsRef.current.delete(job.id)
      setJob(null)
      setJobs((previous) => previous.filter((item) => item.id !== job.id))
      await refreshStatus()
    } catch (caught) {
      const message = displayError(caught)
      if (message) setError(message)
    } finally {
      if (mutationAbort.current === controller) {
        mutationAbort.current = null
        setMutating(false)
      }
    }
  }

  const releaseGpu = async (): Promise<boolean> => {
    if (disabled || active || mutating) return false
    mutationAbort.current?.abort()
    const controller = new AbortController()
    mutationAbort.current = controller
    setMutating(true)
    setError('')
    try {
      const nextStatus = await unloadAndVerifyImageWorker(apiKey, controller.signal)
      setStatus(nextStatus)
      return true
    } catch (caught) {
      const message = displayError(caught)
      if (message) setError(message)
      return false
    } finally {
      if (mutationAbort.current === controller) {
        mutationAbort.current = null
        setMutating(false)
      }
    }
  }

  const useResult = async () => {
    if (!onUseResult || !job || job.status !== 'succeeded' || disabled || mutating) return
    if (!(await releaseGpu())) return
    const controller = new AbortController()
    mutationAbort.current = controller
    setMutating(true)
    try {
      const blob = await imageGenerationApi.imageBlob(job.id, apiKey, controller.signal)
      await onUseResult(blob, inMemoryPromptsRef.current.get(job.id) ?? '')
    } catch (caught) {
      const message = displayError(caught)
      if (message) setError(message)
    } finally {
      if (mutationAbort.current === controller) {
        mutationAbort.current = null
        setMutating(false)
      }
    }
  }

  const ready = status?.enabled === true && status.available

  return (
    <details
      className="image-generation-panel"
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        <span className="image-generation-panel__icon"><ImageIcon size={18} /></span>
        <span>
          <strong>Local image studio</strong>
          <small>Z-Image-Turbo · optional · one GPU</small>
        </span>
        <span className={`image-generation-panel__dot ${ready ? 'is-ready' : ''}`} aria-hidden="true" />
      </summary>

      <div className="image-generation-panel__body">
        <div className="image-generation-panel__status" role="status">
          <span>{describeImageGenerationStatus(status)}</span>
          <span className="image-generation-panel__status-actions">
            {status?.worker_running && (
              <button
                type="button"
                className="image-generation-panel__release"
                disabled={disabled || active || mutating}
                onClick={() => { void releaseGpu() }}
              >
                Release GPU
              </button>
            )}
            <button
              type="button"
              className="image-generation-panel__icon-button"
              aria-label="Refresh image generation status"
              disabled={disabled || statusBusy}
              onClick={() => void refreshStatus()}
            >
              <RefreshCw size={15} className={statusBusy ? 'spin' : ''} />
            </button>
          </span>
        </div>

        <form onSubmit={submit} className="image-generation-panel__form">
          <label>
            <span>Prompt</span>
            <textarea
              value={prompt}
              maxLength={2000}
              rows={4}
              placeholder="Describe a bright, detailed scene…"
              disabled={disabled || !ready || active || mutating}
              onChange={(event) => setPrompt(event.currentTarget.value)}
            />
          </label>

          <div className="image-generation-panel__row">
            <label>
              <span>Canvas</span>
              <select
                value={preset}
                disabled={disabled || !ready || active || mutating}
                onChange={(event) => setPreset(Number(event.currentTarget.value) as ImagePreset)}
              >
                <option value={512}>512 × 512 · quick</option>
                <option value={768}>768 × 768</option>
                <option value={1024}>1024 × 1024 · full</option>
              </select>
            </label>
            <label>
              <span>Seed</span>
              <input
                value={seed}
                inputMode="numeric"
                pattern="[0-9]+"
                maxLength={10}
                disabled={disabled || !ready || active || mutating}
                onChange={(event) => setSeed(event.currentTarget.value)}
              />
            </label>
          </div>

          <label>
            <span>Local API key</span>
            <input
              type="password"
              value={apiKey}
              autoComplete="off"
              maxLength={512}
              placeholder="local-dev-key"
              disabled={disabled || active || mutating}
              onChange={(event) => setApiKey(event.currentTarget.value)}
            />
            <small>Held only in this component's memory; never saved to browser storage.</small>
          </label>

          <button
            className="image-generation-panel__generate"
            type="submit"
            disabled={disabled || !ready || active || mutating || !prompt.trim()}
          >
            {active || mutating
              ? <LoaderCircle className="spin" size={17} />
              : <Sparkles size={17} />}
            {active ? 'Generating locally…' : 'Generate image'}
          </button>
        </form>

        {jobs.length > 0 && (
          <label className="image-generation-panel__library">
            <span>Recent image jobs · {jobs.length}</span>
            <select
              aria-label="Saved image outputs"
              value={job?.id ?? ''}
              disabled={disabled || active || mutating}
              onChange={(event) => {
                const selected = jobs.find((candidate) => candidate.id === event.currentTarget.value)
                if (selected) setJob(selected)
              }}
            >
              {jobs.map((candidate) => (
                <option key={candidate.id} value={candidate.id}>
                  {candidate.status} · {candidate.width}×{candidate.height} · {new Date(candidate.created_at * 1000).toLocaleString()}
                </option>
              ))}
            </select>
            <small>Successful outputs survive restarts. Select any job to inspect or delete it.</small>
          </label>
        )}

        {job && (
          <section className="image-generation-panel__result" aria-live="polite">
            <header>
              <span>
                <strong>{job.status === 'succeeded' ? 'Image ready' : `Job ${job.status}`}</strong>
                <small>
                  {job.width} × {job.height} · {job.settings_known ? `seed ${job.seed}` : 'legacy settings unavailable'}
                </small>
              </span>
              <button
                type="button"
                className="image-generation-panel__icon-button"
                aria-label={active ? 'Cancel and delete image job' : 'Delete image job'}
                disabled={disabled || mutating}
                onClick={() => void removeJob()}
              >
                {active ? <X size={16} /> : <Trash2 size={16} />}
              </button>
            </header>
            {job.status === 'succeeded' && (
              <>
                {previewBusy && <p className="image-generation-panel__preview-status"><LoaderCircle className="spin" size={15} /> Loading private preview…</p>}
                {previewUrl && (
                  <>
                    <img
                      src={previewUrl}
                      alt={inMemoryPromptsRef.current.get(job.id)
                        ? `Locally generated: ${inMemoryPromptsRef.current.get(job.id)!.slice(0, 160)}`
                        : 'Locally generated image'}
                    />
                    <a
                      href={previewUrl}
                      download={`${job.id}.${job.output_format === 'jpeg' ? 'jpg' : 'png'}`}
                    >
                      <Download size={16} /> Download {job.output_format === 'jpeg' ? 'JPEG' : 'PNG'}
                    </a>
                  </>
                )}
                {onUseResult && (
                  <button
                    type="button"
                    className="image-generation-panel__use"
                    disabled={disabled || mutating}
                    onClick={() => void useResult()}
                  >
                    <Sparkles size={16} /> Use in chat · release GPU
                  </button>
                )}
              </>
            )}
            {job.error && <p className="image-generation-panel__error">{job.error}</p>}
          </section>
        )}
        {error && <p className="image-generation-panel__error" role="alert">{error}</p>}
      </div>
    </details>
  )
}

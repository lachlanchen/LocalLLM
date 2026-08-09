import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  imageGenerationApi,
  isTerminalImageJob,
  unloadAndVerifyImageWorker,
} from './imageGenerationApi'

const job = {
  id: 'img_0123456789abcdef0123456789abcdef',
  status: 'queued',
  created_at: 1,
  started_at: null,
  completed_at: null,
  width: 512,
  height: 512,
  steps: 9,
  seed: 42,
  output_format: 'png',
  image_url: null,
  error: null,
  duration_ms: null,
  peak_gpu_memory_bytes: null,
  settings_known: true,
}

const status = {
  enabled: true,
  available: true,
  runtime_ready: true,
  model_ready: true,
  gpu_ready: true,
  gpu_capacity_ready: true,
  gpu_free_memory_bytes: 24 * 1024 ** 3,
  minimum_gpu_free_memory_bytes: 22 * 1024 ** 3,
  worker_running: false,
  model: { id: 'model', revision: 'revision', license: 'Apache-2.0', parameters: 6 },
  gpu: 0,
  limits: {
    concurrency: 1,
    pending_jobs: 4,
    timeout_seconds: 300,
    output_quota_bytes: 1024,
    max_output_files: 128,
    max_output_bytes: 32,
    idle_unload_seconds: 120,
  },
  usage: { queued: 0, running: 0, output_files: 1, output_bytes: 10 },
}

afterEach(() => vi.unstubAllGlobals())

describe('image generation API', () => {
  it('keeps the API key in the request header and sends no remote input URL', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(job), { status: 202 }))
    vi.stubGlobal('fetch', fetchMock)

    await imageGenerationApi.create({
      prompt: 'bright robot',
      width: 512,
      height: 512,
      steps: 9,
      seed: 42,
      output_format: 'png',
      jpeg_quality: 90,
    }, 'memory-only-key')

    const [, init] = fetchMock.mock.calls[0]
    expect(init.headers['X-LocalLLM-Key']).toBe('memory-only-key')
    expect(init.body).not.toContain('image_url')
    expect(init.body).not.toContain('memory-only-key')
  })

  it('supports cancellation/delete with a bounded local job ID', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
    vi.stubGlobal('fetch', fetchMock)

    await imageGenerationApi.delete(job.id, 'key')

    expect(fetchMock).toHaveBeenCalledWith(`/api/images/jobs/${job.id}`, expect.objectContaining({
      method: 'DELETE',
    }))
    await expect(imageGenerationApi.job('../../etc/passwd', 'key')).rejects.toThrow(
      'Invalid local image job ID',
    )
  })

  it('loads a bounded persisted job list', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify([
      { ...job, status: 'succeeded', image_url: `/api/images/jobs/${job.id}/image` },
    ]), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const jobs = await imageGenerationApi.jobs('private-key')

    expect(jobs).toHaveLength(1)
    expect(jobs[0]).toMatchObject({ id: job.id, settings_known: true })
    expect(fetchMock).toHaveBeenCalledWith('/api/images/jobs', {
      headers: { 'X-LocalLLM-Key': 'private-key' },
    })
  })

  it('downloads retained images with authentication and bounded media validation', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(new Blob(['png'], {
      type: 'image/png',
    }), {
      status: 200,
      headers: { 'Content-Type': 'image/png', 'Content-Length': '3' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    const blob = await imageGenerationApi.imageBlob(job.id, 'private-key')

    expect(blob.type).toBe('image/png')
    expect(fetchMock).toHaveBeenCalledWith(`/api/images/jobs/${job.id}/image`, {
      headers: { 'X-LocalLLM-Key': 'private-key' },
    })
  })

  it('releases warm GPU weights through an authenticated empty request', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      released: true,
      worker_running: false,
    }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(await imageGenerationApi.unload('memory-key')).toBe(true)
    expect(fetchMock).toHaveBeenCalledWith('/api/images/unload', expect.objectContaining({
      method: 'POST',
      headers: { 'X-LocalLLM-Key': 'memory-key' },
    }))
  })

  it('does not verify release after an unload authentication failure', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: 'Invalid LocalLLM API key',
    }), { status: 401 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(unloadAndVerifyImageWorker('bad-key')).rejects.toThrow('Invalid LocalLLM API key')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('fails release verification when the worker remains resident', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        released: true,
        worker_running: false,
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        ...status,
        worker_running: true,
      }), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(unloadAndVerifyImageWorker('private-key')).rejects.toThrow('did not release')
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('recognizes all terminal states', () => {
    expect(isTerminalImageJob('queued')).toBe(false)
    expect(isTerminalImageJob('running')).toBe(false)
    expect(isTerminalImageJob('succeeded')).toBe(true)
    expect(isTerminalImageJob('failed')).toBe(true)
    expect(isTerminalImageJob('cancelled')).toBe(true)
  })
})

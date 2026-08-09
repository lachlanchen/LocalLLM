import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import {
  describeImageGenerationStatus,
  imageGenerationPanelIsBusy,
  ImageGenerationPanel,
} from './ImageGenerationPanel'

describe('image generation panel', () => {
  it('is collapsed by default and preloads only the public local interoperability key', () => {
    const html = renderToStaticMarkup(<ImageGenerationPanel />)

    expect(html).toContain('<details class="image-generation-panel">')
    expect(html).not.toContain('<details class="image-generation-panel" open=""')
    expect(html).toContain('placeholder="local-dev-key"')
    expect(html).toContain('value="local-dev-key"')
    expect(html).toContain('Local image studio')
  })

  it('supports a disabled parent state and an explicit use-result callback', () => {
    const html = renderToStaticMarkup(
      <ImageGenerationPanel disabled onUseResult={() => undefined} />,
    )

    expect(html).toContain('disabled=""')
    expect(html).toContain('value="local-dev-key"')
  })

  it('explains disabled, missing-runtime, missing-model, and ready states', () => {
    const base = {
      enabled: false,
      available: false,
      runtime_ready: false,
      model_ready: false,
      gpu_ready: true,
      gpu_capacity_ready: true,
      gpu_free_memory_bytes: 24 * 1024 ** 3,
      minimum_gpu_free_memory_bytes: 22 * 1024 ** 3,
      worker_running: false,
      model: { id: 'model', revision: 'revision', license: 'Apache-2.0', parameters: 6 },
      gpu: 1,
      limits: {
        concurrency: 1,
        pending_jobs: 4,
        timeout_seconds: 300,
        output_quota_bytes: 1,
        max_output_files: 1,
        max_output_bytes: 1,
        idle_unload_seconds: 120,
      },
      usage: { queued: 0, running: 0, output_files: 0, output_bytes: 0 },
    }

    expect(describeImageGenerationStatus(base)).toContain('Disabled')
    expect(describeImageGenerationStatus({ ...base, enabled: true })).toContain('runtime')
    expect(describeImageGenerationStatus({
      ...base, enabled: true, runtime_ready: true,
    })).toContain('not downloaded')
    expect(describeImageGenerationStatus({
      ...base,
      enabled: true,
      available: true,
      runtime_ready: true,
      model_ready: true,
      gpu_ready: true,
      gpu_capacity_ready: true,
    })).toContain('Ready on GPU 1')
    expect(describeImageGenerationStatus({
      ...base,
      enabled: true,
      runtime_ready: true,
      model_ready: true,
      gpu_capacity_ready: false,
      gpu_free_memory_bytes: 7 * 1024 ** 3,
    })).toContain('at least 22 GiB')
  })

  it('keeps the parent busy through active and warm-worker lifecycle states', () => {
    const queued = {
      id: 'img_0123456789abcdef0123456789abcdef',
      status: 'queued' as const,
      created_at: 1,
      started_at: null,
      completed_at: null,
      width: 512,
      height: 512,
      steps: 9,
      seed: 1,
      output_format: 'png' as const,
      image_url: null,
      error: null,
      duration_ms: null,
      peak_gpu_memory_bytes: null,
      settings_known: true,
    }
    const warmStatus = {
      enabled: true,
      available: true,
      runtime_ready: true,
      model_ready: true,
      gpu_ready: true,
      gpu_capacity_ready: true,
      gpu_free_memory_bytes: 24 * 1024 ** 3,
      minimum_gpu_free_memory_bytes: 22 * 1024 ** 3,
      worker_running: true,
      model: { id: 'model', revision: 'revision', license: 'Apache-2.0', parameters: 6 },
      gpu: 1,
      limits: {
        concurrency: 1,
        pending_jobs: 4,
        timeout_seconds: 300,
        output_quota_bytes: 1,
        max_output_files: 1,
        max_output_bytes: 1,
        idle_unload_seconds: 120,
      },
      usage: { queued: 0, running: 0, output_files: 0, output_bytes: 0 },
    }

    expect(imageGenerationPanelIsBusy(null, queued, false)).toBe(true)
    expect(imageGenerationPanelIsBusy(warmStatus, { ...queued, status: 'succeeded' }, false)).toBe(true)
    expect(imageGenerationPanelIsBusy({ ...warmStatus, worker_running: false }, null, false)).toBe(false)
    expect(imageGenerationPanelIsBusy(null, null, true)).toBe(true)
  })
})

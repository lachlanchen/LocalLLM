import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { AppHeader, ModelCard } from './App'
import type { ModelInfo, SystemStatus } from './types'

const status: SystemStatus = {
  service: { ok: true, version: 'test' },
  gpu: { ok: true, backend: 'metal', devices: [{ index: 0, name: 'Apple GPU' }] },
  ollama: { ok: true }, storage: { total: 0, used: 0, free: 0 },
  binding: { host: '127.0.0.1', port: 8008, local_only: true },
}

describe('Mac hardware and configured model context', () => {
  it('shows observed Metal hardware without claiming model inference readiness', () => {
    const html = renderToStaticMarkup(<AppHeader status={status} onRefresh={() => {}} />)
    expect(html).toContain('Metal supported')
    expect(html).not.toContain('GPU needs attention')
    expect(html).not.toContain('GPU ready')
  })

  it('keeps failed Metal detection visibly unavailable', () => {
    const unavailable = { ...status, gpu: { ...status.gpu, ok: false, devices: [] } }
    expect(renderToStaticMarkup(<AppHeader status={unavailable} onRefresh={() => {}} />))
      .toContain('Metal status unavailable')
  })

  it('shows the app context limit alongside the model maximum', () => {
    const model: ModelInfo = {
      id: 'qwen3.6:35b-a3b', family: 'Qwen3.6', quantization: '4-bit', size_gb: 24,
      context: 262144, runtime_context: 8192, modalities: ['text', 'image'],
      tier: 'Qwen3.6', role: 'Chat', installed: true, recommended: false,
    }
    const html = renderToStaticMarkup(<ModelCard model={model} onPull={() => {}} />)
    expect(html).toContain('8K</strong><span>APP CONTEXT')
    expect(html).toContain('Model maximum: 256K; app limit: 8K')
    expect(html).toContain('Installed')
  })
})

import { describe, expect, it } from 'vitest'
import { chooseAvailableAlias, isAliasInstalled, modelChoiceStatuses } from './modelAvailability'
import type { CatalogResponse, ModelInfo } from './types'

const models: ModelInfo[] = [
  {
    id: 'qwen3:4b-q4_K_M',
    family: 'Qwen3 4B',
    quantization: 'Q4_K_M',
    size_gb: 2.6,
    context: 40960,
    modalities: ['text'],
    tier: 'Pocket',
    role: 'Fast experiments',
    recommended: false,
    installed: true,
  },
  {
    id: 'qwen3:8b-q4_K_M',
    family: 'Qwen3 8B',
    quantization: 'Q4_K_M',
    size_gb: 5.2,
    context: 40960,
    modalities: ['text', 'tools'],
    tier: 'Fast',
    role: 'Everyday assistant',
    recommended: true,
    installed: false,
  },
  {
    id: 'qwen3-vl:8b-instruct-q8_0',
    family: 'Qwen3-VL 8B',
    quantization: 'Q8_0',
    size_gb: 9.8,
    context: 262144,
    modalities: ['text', 'image'],
    tier: 'Vision+',
    role: 'Detailed vision',
    recommended: false,
    installed: true,
  },
]

const catalog: CatalogResponse = {
  models,
  installed: [],
  aliases: {
    'localllm-pocket': 'qwen3:4b-q4_K_M',
    'localllm-fast': 'qwen3:8b-q4_K_M',
    'localllm-balanced': 'qwen3:8b-q8_0',
    'localllm-deep': 'qwen3:30b-a3b-instruct-2507-q4_K_M',
    'localllm-max': 'qwen3:30b-a3b-instruct-2507-q8_0',
    'localllm-vision': 'qwen3-vl:8b-instruct-q4_K_M',
    'localllm-vision-max': 'qwen3-vl:8b-instruct-q8_0',
    'localllm-vision-xl': 'qwen3-vl:30b-a3b-instruct-q4_K_M',
  },
  planned_download_gb: 109.2,
}

describe('local model availability', () => {
  it('keeps an installed preferred alias', () => {
    expect(chooseAvailableAlias(catalog, 'localllm-pocket', 'text')).toBe('localllm-pocket')
  })

  it('falls back to an installed model while the default is downloading', () => {
    expect(chooseAvailableAlias(catalog, 'localllm-fast', 'text')).toBe('localllm-pocket')
  })

  it('accepts exact curated IDs for models without a stable alias', () => {
    const q8Only = {
      ...catalog,
      models: catalog.models.map((model) => ({
        ...model,
        installed: model.id === 'qwen3:4b-q4_K_M' ? false : model.installed,
      })).concat({ ...models[0], id: 'qwen3:4b-q8_0', quantization: 'Q8_0', installed: true }),
    }
    expect(chooseAvailableAlias(q8Only, 'localllm-fast', 'text')).toBe('qwen3:4b-q8_0')
  })

  it('uses an installed vision quantization without falling back to text', () => {
    expect(chooseAvailableAlias(catalog, 'localllm-vision', 'vision')).toBe('localllm-vision-max')
    expect(isAliasInstalled(catalog, 'localllm-vision-max')).toBe(true)
  })

  it('returns no choice when a modality has no installed model', () => {
    const withoutVision = { ...catalog, models: catalog.models.filter((model) => !model.modalities.includes('image')) }
    expect(chooseAvailableAlias(withoutVision, 'localllm-vision', 'vision')).toBeNull()
  })

  it('distinguishes an unchecked catalog from a checked missing model', () => {
    expect(modelChoiceStatuses(null, 'text')[0].installed).toBeNull()
    expect(modelChoiceStatuses(catalog, 'text')[0].installed).toBe(false)
  })
})

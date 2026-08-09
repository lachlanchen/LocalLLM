import type { CatalogResponse } from './types'

export type ModelKind = 'text' | 'vision'

export interface ModelChoice {
  alias: string
  label: string
}

export interface ModelChoiceStatus extends ModelChoice {
  installed: boolean | null
}

export const TEXT_MODEL_CHOICES: readonly ModelChoice[] = [
  { alias: 'localllm-fast', label: 'Qwen3 8B · Fast' },
  { alias: 'localllm-balanced', label: 'Qwen3 8B · Q8' },
  { alias: 'localllm-deep', label: 'Qwen3 30B · Deep' },
  { alias: 'localllm-max', label: 'Qwen3 30B · Q8' },
  { alias: 'localllm-pocket', label: 'Qwen3 4B · Pocket' },
  { alias: 'qwen3:4b-q8_0', label: 'Qwen3 4B · Q8' },
]

export const VISION_MODEL_CHOICES: readonly ModelChoice[] = [
  { alias: 'localllm-vision', label: 'Qwen3-VL 8B · Q4' },
  { alias: 'localllm-vision-max', label: 'Qwen3-VL 8B · Q8' },
  { alias: 'localllm-vision-xl', label: 'Qwen3-VL 30B · XL' },
]

export function choicesFor(kind: ModelKind): readonly ModelChoice[] {
  return kind === 'vision' ? VISION_MODEL_CHOICES : TEXT_MODEL_CHOICES
}

export function isAliasInstalled(catalog: CatalogResponse | null, alias: string): boolean {
  if (!catalog) return false
  const target = catalog.aliases[alias] ?? alias
  return Boolean(target && catalog.models.some((model) => model.id === target && model.installed))
}

export function modelChoiceStatuses(catalog: CatalogResponse | null, kind: ModelKind): ModelChoiceStatus[] {
  return choicesFor(kind).map((choice) => ({
    ...choice,
    installed: catalog ? isAliasInstalled(catalog, choice.alias) : null,
  }))
}

/** Keep a usable choice selected as partial model downloads finish. */
export function chooseAvailableAlias(
  catalog: CatalogResponse | null,
  preferred: string,
  kind: ModelKind,
): string | null {
  if (!catalog) return null
  if (isAliasInstalled(catalog, preferred)) return preferred
  return choicesFor(kind).find((choice) => isAliasInstalled(catalog, choice.alias))?.alias ?? null
}

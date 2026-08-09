export const AGENT_AUTO_STORAGE_KEY = 'localllm.agent-routing.enabled.v1'
export const MAX_AGENT_GOAL_CHARS = 4_000

export interface AgentPreferenceStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
}

/**
 * Agent routing is intentionally opt-out. A missing, inaccessible, or malformed
 * preference therefore uses the safe product default without blocking chat.
 */
export function readAgentAutoEnabled(storage?: AgentPreferenceStorage | null): boolean {
  try {
    const target = storage === undefined ? globalThis.localStorage : storage
    const stored = target?.getItem(AGENT_AUTO_STORAGE_KEY)
    if (stored === 'false') return false
    if (stored === 'true') return true
  } catch {
    // localStorage can be unavailable or throw under restrictive browser policies.
  }
  return true
}

/** Store only the boolean preference; conversation and task content never enter storage. */
export function writeAgentAutoEnabled(
  enabled: boolean,
  storage?: AgentPreferenceStorage | null,
): boolean {
  try {
    const target = storage === undefined ? globalThis.localStorage : storage
    if (!target) return false
    target.setItem(AGENT_AUTO_STORAGE_KEY, enabled ? 'true' : 'false')
    return true
  } catch {
    return false
  }
}

const ENGLISH_NEGATED_EXECUTION = [
  /\b(?:do\s+not|don't|dont|never|avoid|without)\s+(?:actually\s+)?(?:run|execute|evaluate)\b/,
  /\bwithout\s+(?:running|executing|evaluating)\b/,
  /\b(?:not|never)\s+to\s+(?:run|execute|evaluate)\b/,
]

const ENGLISH_HOW_TO_OR_HYPOTHETICAL = [
  /\b(?:how\s+(?:do|can|could|would|should)\s+(?:i|we|you|one)|how\s+to|show\s+me\s+how\s+to)\b[^.!?\n]{0,100}\b(?:run|execute|evaluate)\b/,
  /\b(?:what|which)[^.!?\n]{0,100}\bif\b[^.!?\n]{0,50}\b(?:run|execute|evaluate)\b/,
  /\b(?:should|could|would)\s+(?:i|we)\s+(?:run|execute|evaluate)\b/,
  /\b(?:can|could|would|should|will)\s+(?:this|that|the)\s+(?:python\s+)?(?:code|snippet|script|program)\s+(?:run|execute)\b/,
]

const NON_PYTHON_RUNTIME = /\b(?:bash|shell|zsh|fish|powershell|pwsh|cmd(?:\.exe)?|javascript|typescript|node(?:\.js)?|npm|npx|sql|sqlite|postgres(?:ql)?|mysql|rust|golang|java|kotlin|swift|ruby|php|perl|matlab|julia|c\+\+|c#)\b/
const NON_PYTHON_FENCE = /^\s*```\s*(?:bash|sh|zsh|fish|powershell|pwsh|javascript|js|typescript|ts|sql|rust|go|java|kotlin|swift|ruby|php|perl|r|matlab|julia|c|cpp|csharp)\b/im

const DIRECT_ENGLISH_EXECUTION = [
  /\b(?:run|execute|evaluate)\s+(?:(?:the|this|that|following|below|attached|given|my|some)\s+)?(?:python\s+)?(?:code|snippet|script|program)\b/,
  /\buse\s+python\s+to\s+(?:calculate|compute|evaluate|solve|process|parse|convert|print|plot|analy[sz]e|check|verify)\b/,
]

const DIRECT_CJK_EXECUTION = [
  /(?:运行|運行|执行|執行|跑一下|跑下)\s*(?:这|這|该|該|以下|下面|下列|此|我的)?\s*的?\s*(?:一?段)?\s*(?:python\s*)?(?:代码|代碼|脚本|腳本|程式|程序)/,
  /(?:把|将|將|请|請)?[^。！？\n]{0,18}(?:python\s*)?(?:代码|代碼|脚本|腳本|程式|程序)[^。！？\n]{0,12}(?:运行|運行|执行|執行|跑一下|跑下)/,
  /用\s*python\s*(?:来|來)?\s*(?:计算|計算|运算|運算|求解|处理|處理|解析|转换|轉換|生成|打印|输出|輸出|绘制|繪製|检查|檢查|验证|驗證)/,
]

function withoutAuthoredCodeOrQuotes(value: string): string {
  return value
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`\n]*`/g, ' ')
    .replace(/"[^"\n]{1,240}"/g, ' ')
    .replace(/“[^”\n]{1,240}”/g, ' ')
    .replace(/‘[^’\n]{1,240}’/g, ' ')
    .replace(/(^|[\s([{:,])'[^'\n]{1,240}'(?=$|[\s)\]},.!?;:])/g, '$1 ')
}

/**
 * Return true only for an explicit request to execute code in the isolated
 * Python tool. This is routing, not authorization: the Agent flow still stages,
 * previews, and confirms generated code before execution.
 */
export function shouldAutoRouteAgent(goal: string): boolean {
  if (typeof goal !== 'string') return false
  const trimmed = goal.trim()
  if (!trimmed) return false

  const normalized = trimmed.normalize('NFKC').toLowerCase()
  const prose = withoutAuthoredCodeOrQuotes(normalized).replace(/\s+/g, ' ').trim()
  if (!prose) return false

  const pythonIsNegated = /\b(?:do\s+not|don't|dont|never|avoid|without)\s+(?:use|using)?\s*python\b|\bnot\s+python\b/.test(prose)
  const explicitlyPython = /\bpython\b/.test(prose) && !pythonIsNegated
  if ((NON_PYTHON_RUNTIME.test(prose) && !explicitlyPython) || NON_PYTHON_FENCE.test(normalized)) {
    return false
  }
  if (ENGLISH_NEGATED_EXECUTION.some((pattern) => pattern.test(prose))) return false
  if (ENGLISH_HOW_TO_OR_HYPOTHETICAL.some((pattern) => pattern.test(prose))) return false
  if (/(?:不要|不必|无需|無需|不用|别|別|禁止)[^。！？\n]{0,12}(?:运行|運行|执行|執行|跑)/.test(prose)) {
    return false
  }
  if (/(?:如何|怎么|怎麼|怎样|怎樣)[^。！？\n]{0,20}(?:运行|運行|执行|執行|跑)/.test(prose)) {
    return false
  }
  if (/\b(?:run|execute)\s+(?:(?:a|the|this|that)\s+)?(?:code\s+)?(?:plan|workflow|strategy|review|audit|analysis|inspection|quality\s+check)\b/.test(prose)) {
    return false
  }

  if (DIRECT_ENGLISH_EXECUTION.some((pattern) => pattern.test(prose))) return true
  if (DIRECT_CJK_EXECUTION.some((pattern) => pattern.test(prose))) return true

  // A code block plus a direct deictic command is still explicit. Instructions
  // merely embedded inside the code block were removed from `prose` above.
  const hasCodeFence = /```[\s\S]*?```/.test(normalized)
  return hasCodeFence && /\b(?:run|execute|evaluate)\s+(?:this|that|it|the\s+following|the\s+code)\b/.test(prose)
}

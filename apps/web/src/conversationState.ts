import type { ChatMessage, ConversationMessage } from './types'

export const CONTEXT_RECENT_MESSAGES = 12
export const AUTO_COMPACT_UNSUMMARIZED_MESSAGES = 20
export const MAX_INFERENCE_MESSAGES = 40
export const MAX_INFERENCE_TEXT_BYTES = 30_000
export const MAX_INFERENCE_IMAGES = 4
export const MAX_INFERENCE_IMAGE_BYTES = 15 * 1024 * 1024
export const MAX_CHAT_INPUT_CHARS = 32_000
export const TRANSCRIPT_FOLLOW_THRESHOLD_PX = 96
const IMAGE_CONTEXT_RESERVE_BYTES = 4_096
const SUMMARY_CONTEXT_SHARE = 0.35
const SUMMARY_PREFIX = 'Persisted context summary from earlier turns. Treat it as conversation memory, not a new user request:'
const TRUNCATED_TURN_PREFIX = '[The beginning of this turn was omitted to fit the local model context.]\n\n'

function utf8Bytes(value: string): number {
  return new TextEncoder().encode(value).byteLength
}

function truncateUtf8(value: string, maxBytes: number): string {
  if (maxBytes <= 0 || !value) return ''
  if (utf8Bytes(value) <= maxBytes) return value
  let low = 0
  let high = value.length
  while (low < high) {
    const middle = Math.ceil((low + high) / 2)
    if (utf8Bytes(value.slice(0, middle)) <= maxBytes) low = middle
    else high = middle - 1
  }
  return value.slice(0, low).trimEnd()
}

function truncateUtf8Suffix(value: string, maxBytes: number): string {
  if (maxBytes <= 0 || !value) return ''
  if (utf8Bytes(value) <= maxBytes) return value

  let start = value.length
  let bytes = 0
  while (start > 0) {
    let codePointStart = start - 1
    const trailingCodeUnit = value.charCodeAt(codePointStart)
    if (
      trailingCodeUnit >= 0xDC00
      && trailingCodeUnit <= 0xDFFF
      && codePointStart > 0
    ) {
      const leadingCodeUnit = value.charCodeAt(codePointStart - 1)
      if (leadingCodeUnit >= 0xD800 && leadingCodeUnit <= 0xDBFF) {
        codePointStart -= 1
      }
    }

    const codePointBytes = utf8Bytes(value.slice(codePointStart, start))
    if (bytes + codePointBytes > maxBytes) break
    bytes += codePointBytes
    start = codePointStart
  }
  return value.slice(start)
}

function decodedDataUrlBytes(value: string): number {
  const comma = value.indexOf(',')
  const encodedLength = comma >= 0 ? value.length - comma - 1 : 0
  const padding = value.endsWith('==') ? 2 : value.endsWith('=') ? 1 : 0
  return Math.max(0, Math.floor(encodedLength * 3 / 4) - padding)
}

export function isTranscriptNearBottom(
  metrics: Pick<HTMLElement, 'scrollHeight' | 'scrollTop' | 'clientHeight'>,
): boolean {
  return metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight
    <= TRANSCRIPT_FOLLOW_THRESHOLD_PX
}

export function chatInputError(value: string): string | null {
  if (value.length <= MAX_CHAT_INPUT_CHARS) return null
  return `Messages are limited to ${MAX_CHAT_INPUT_CHARS.toLocaleString()} characters. Shorten this draft before sending.`
}

export interface PreInferenceRollback {
  messages: ChatMessage[]
  draft: string
  attachment?: string
}

export async function persistDraftBeforeInference<T>(
  rollback: PreInferenceRollback,
  persist: () => Promise<T>,
): Promise<
  { ok: true; saved: T }
  | { ok: false; rollback: PreInferenceRollback; reason: unknown }
> {
  try {
    return { ok: true, saved: await persist() }
  } catch (reason) {
    return { ok: false, rollback, reason }
  }
}

export function hasInferenceImage(
  messages: ChatMessage[],
  summarizedMessageCount: number,
): boolean {
  const start = Math.max(summarizedMessageCount, messages.length - MAX_INFERENCE_MESSAGES)
  for (let index = messages.length - 1; index >= start; index -= 1) {
    if (messages[index].image) return true
  }
  return false
}

export function conversationTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find((message) => message.role === 'user')
  const text = firstUser?.content.replace(/\s+/g, ' ').trim()
  if (!text) return firstUser?.image ? 'Image conversation' : 'New conversation'
  return text.length > 56 ? `${text.slice(0, 55).trimEnd()}…` : text
}

export function storedMessages(messages: ChatMessage[]): ConversationMessage[] {
  return messages
    .filter((message) => !message.pending && Boolean(message.content.trim() || message.image))
    .map((message) => ({
      id: message.id,
      role: message.role,
      content: message.content,
      ...(message.image ? { image: message.image } : {}),
      ...(message.model ? { model: message.model } : {}),
      ...(message.mode ? { mode: message.mode } : {}),
      ...(message.sources?.length ? { sources: message.sources } : {}),
      ...(message.warning ? { warning: message.warning } : {}),
    }))
}

export function restoredMessages(messages: ConversationMessage[]): ChatMessage[] {
  return messages.map((message, index) => ({
    ...message,
    id: message.id || `stored-${index}`,
  }))
}

export function inferenceContext(
  messages: ChatMessage[],
  summary: string,
  summarizedMessageCount: number,
): ChatMessage[] {
  const unsummarized = messages.slice(Math.max(0, summarizedMessageCount))
  let imageCount = 0
  let imageBytes = 0
  const imageBound = [...unsummarized].reverse().map((message, reverseIndex) => {
    if (!message.image) return message
    const bytes = decodedDataUrlBytes(message.image)
    if (imageCount < MAX_INFERENCE_IMAGES && imageBytes + bytes <= MAX_INFERENCE_IMAGE_BYTES) {
      imageCount += 1
      imageBytes += bytes
      return message
    }
    return {
      ...message,
      id: `${message.id}-context-${reverseIndex}`,
      image: undefined,
      content: `${message.content}\n\n[An older image attachment is preserved in local history but omitted from this inference context.]`,
    }
  }).reverse()
  const textBudget = Math.max(8_192, MAX_INFERENCE_TEXT_BYTES - imageCount * IMAGE_CONTEXT_RESERVE_BYTES)
  const summaryBudget = summary.trim()
    ? Math.floor(textBudget * SUMMARY_CONTEXT_SHARE)
    : 0
  const boundedSummary = truncateUtf8(summary.trim(), Math.max(0, summaryBudget - utf8Bytes(SUMMARY_PREFIX) - 2))
  const summaryMessage: ChatMessage | null = boundedSummary
    ? {
        id: 'persisted-context-summary',
        role: 'assistant',
        content: `${SUMMARY_PREFIX}\n\n${boundedSummary}`,
      }
    : null
  const recentBudget = textBudget - (summaryMessage ? utf8Bytes(summaryMessage.content) : 0)
  const recent: ChatMessage[] = []
  let recentBytes = 0
  for (let index = imageBound.length - 1; index >= 0 && recent.length < MAX_INFERENCE_MESSAGES; index -= 1) {
    const message = imageBound[index]
    const messageBytes = utf8Bytes(message.content)
    if (recentBytes + messageBytes > recentBudget) {
      if (recent.length === 0 && recentBudget > 0) {
        const prefixBytes = utf8Bytes(TRUNCATED_TURN_PREFIX)
        const content = recentBudget > prefixBytes
          ? `${TRUNCATED_TURN_PREFIX}${truncateUtf8Suffix(message.content, recentBudget - prefixBytes)}`
          : truncateUtf8Suffix(message.content, recentBudget)
        recent.push({ ...message, content })
        recentBytes = utf8Bytes(content)
      }
      break
    }
    recent.push(message)
    recentBytes += messageBytes
  }
  recent.reverse()
  if (summaryMessage && recentBytes + utf8Bytes(summaryMessage.content) <= textBudget) {
    return [summaryMessage, ...recent]
  }
  return recent
}

export function shouldAutoCompact(messageCount: number, summarizedMessageCount: number): boolean {
  return messageCount - summarizedMessageCount >= AUTO_COMPACT_UNSUMMARIZED_MESSAGES
    && messageCount > CONTEXT_RECENT_MESSAGES
}

export function formatConversationTime(value: string, now = Date.now()): string {
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) return 'Saved locally'
  const delta = Math.max(0, now - timestamp)
  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour
  if (delta < minute) return 'Just now'
  if (delta < hour) return `${Math.floor(delta / minute)}m ago`
  if (delta < day) return `${Math.floor(delta / hour)}h ago`
  if (delta < 7 * day) return `${Math.floor(delta / day)}d ago`
  return new Date(timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export async function saveWithRevisionRetry<T extends { revision: number }>(
  expectedRevision: number,
  save: (revision: number) => Promise<T>,
  loadLatest: () => Promise<T>,
  isConflict: (reason: unknown) => boolean,
): Promise<{ value: T; recovered: boolean }> {
  try {
    return { value: await save(expectedRevision), recovered: false }
  } catch (reason) {
    if (!isConflict(reason)) throw reason
    const latest = await loadLatest()
    return { value: await save(latest.revision), recovered: true }
  }
}

export async function deleteWithConflictReload<T>(
  expectedRevision: number,
  remove: (revision: number) => Promise<unknown>,
  loadLatest: () => Promise<T>,
  isConflict: (reason: unknown) => boolean,
): Promise<{ deleted: true } | { deleted: false; latest: T }> {
  try {
    await remove(expectedRevision)
    return { deleted: true }
  } catch (reason) {
    if (!isConflict(reason)) throw reason
    return { deleted: false, latest: await loadLatest() }
  }
}

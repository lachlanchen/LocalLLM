import { describe, expect, it } from 'vitest'
import {
  chatInputError,
  conversationTitle,
  deleteWithConflictReload,
  formatConversationTime,
  inferenceContext,
  isTranscriptNearBottom,
  MAX_CHAT_INPUT_CHARS,
  MAX_INFERENCE_TEXT_BYTES,
  persistDraftBeforeInference,
  removeImageAt,
  restoredMessages,
  saveWithRevisionRetry,
  shouldAutoCompact,
  storedMessages,
} from './conversationState'
import type { ChatMessage } from './types'

const messages: ChatMessage[] = [
  { id: 'u1', role: 'user', content: '  Explain   the control loop in detail.  ', mode: 'local' },
  { id: 'a1', role: 'assistant', content: 'A stable answer.', model: 'localllm-fast', mode: 'local' },
]

describe('durable conversation state', () => {
  it('accepts the exact chat-input boundary and rejects one additional character', () => {
    expect(chatInputError('x'.repeat(MAX_CHAT_INPUT_CHARS))).toBeNull()
    expect(chatInputError('x'.repeat(MAX_CHAT_INPUT_CHARS + 1))).toContain('32,000')
  })

  it('returns the exact persisted transcript, draft, and attachment after a rejected pre-save', async () => {
    const persisted = [...messages]
    const draft = '  Keep my exact spacing.\nAnd this newline.  '
    const attachments = [
      'data:image/png;base64,Zmlyc3Q=',
      'data:image/jpeg;base64,c2Vjb25k',
    ]
    const failure = new Error('archive quota reached')
    let attempts = 0

    const rejected = await persistDraftBeforeInference(
      { messages: persisted, draft, attachments },
      async () => {
        attempts += 1
        throw failure
      },
    )

    expect(rejected).toEqual({
      ok: false,
      rollback: { messages: persisted, draft, attachments },
      reason: failure,
    })
    if (!rejected.ok) {
      expect(rejected.rollback.messages).toBe(persisted)
      expect(rejected.rollback.draft).toBe(draft)
      expect(rejected.rollback.attachments).toBe(attachments)
    }

    const retry = await persistDraftBeforeInference(
      { messages: persisted, draft: 'smaller retry', attachments: [] },
      async () => {
        attempts += 1
        return { revision: 9 }
      },
    )
    expect(attempts).toBe(2)
    expect(retry).toEqual({ ok: true, saved: { revision: 9 } })
  })

  it('derives a compact title from the first user turn', () => {
    expect(conversationTitle(messages)).toBe('Explain the control loop in detail.')
    expect(conversationTitle([{ id: 'image', role: 'user', content: '', images: ['data:image/png;base64,AA=='] }])).toBe('Image conversation')
  })

  it('persists only finalized display metadata and restores stable IDs', () => {
    const persisted = storedMessages([
      ...messages,
      { id: 'pending', role: 'assistant', content: '', pending: true, activity: ['thinking'] },
    ])
    expect(persisted).toHaveLength(2)
    expect(persisted[1]).toEqual({
      id: 'a1', role: 'assistant', content: 'A stable answer.', model: 'localllm-fast', mode: 'local',
    })
    expect(restoredMessages([{ role: 'assistant', content: 'Saved' }])[0].id).toBe('stored-0')
    expect(restoredMessages([{
      role: 'user', content: '', image: 'data:image/png;base64,bGVnYWN5',
    }])[0]).toEqual({
      id: 'stored-0', role: 'user', content: '', images: ['data:image/png;base64,bGVnYWN5'],
    })
  })

  it('sends a persisted summary plus only the unsummarized tail', () => {
    const context = inferenceContext(messages, 'The user is building a robot.', 1)
    expect(context).toHaveLength(2)
    expect(context[0].role).toBe('assistant')
    expect(context[0].content).toContain('building a robot')
    expect(context[1].id).toBe('a1')
    expect(inferenceContext(messages, '', 1)).toEqual([messages[1]])
  })

  it('keeps full image history but sends only the newest four images to inference', () => {
    const imageTurns: ChatMessage[] = Array.from({ length: 5 }, (_, index) => ({
      id: `image-${index}`,
      role: 'user' as const,
      content: `image turn ${index}`,
      images: [`data:image/png;base64,${btoa(`image-${index}`)}`],
    }))

    const context = inferenceContext(imageTurns, '', 0)

    expect(imageTurns.filter((message) => message.images?.length)).toHaveLength(5)
    expect(context.filter((message) => message.images?.length)).toHaveLength(4)
    expect(context[0].images).toBeUndefined()
    expect(context[0].content).toContain('preserved in local history')
    expect(context.at(-1)?.images).toBeTruthy()
  })

  it('preserves attachment order and removes exactly the selected thumbnail', () => {
    const ordered = ['first', 'second', 'third', 'fourth']
    const message: ChatMessage = { id: 'ordered', role: 'user', content: 'Compare', images: ordered }

    expect(inferenceContext([message], '', 0)[0].images).toEqual(ordered)
    expect(removeImageAt(ordered, 1)).toEqual(['first', 'third', 'fourth'])
    expect(removeImageAt(ordered, 99)).toEqual(ordered)
  })

  it('bounds multilingual resumed context by UTF-8 bytes including its summary', () => {
    const dense: ChatMessage[] = Array.from({ length: 8 }, (_, index) => ({
      id: `dense-${index}`,
      role: index % 2 ? 'assistant' as const : 'user' as const,
      content: '证据与上下文'.repeat(1_000),
    }))

    const context = inferenceContext(dense, '较早对话摘要'.repeat(2_000), 0)
    const bytes = context.reduce(
      (total, message) => total + new TextEncoder().encode(message.content).byteLength,
      0,
    )

    expect(bytes).toBeLessThanOrEqual(MAX_INFERENCE_TEXT_BYTES)
    expect(context.at(-1)?.id).toBe('dense-7')
  })

  it('strictly bounds an oversized newest turn while preserving its actual tail question', () => {
    const tailQuestion = 'LATEST QUESTION SENTINEL: Which actuator should I tune first?'
    const context = inferenceContext([{
      id: 'oversized-latest',
      role: 'user',
      content: `EARLIEST PREFIX SENTINEL ${'x'.repeat(MAX_INFERENCE_TEXT_BYTES + 1_000)}${tailQuestion}`,
    }], '', 0)
    const bytes = context.reduce(
      (total, message) => total + new TextEncoder().encode(message.content).byteLength,
      0,
    )

    expect(context).toHaveLength(1)
    expect(context[0].id).toBe('oversized-latest')
    expect(context[0].content).toMatch(/^\[The beginning of this turn was omitted to fit/)
    expect(context[0].content).not.toContain('EARLIEST PREFIX SENTINEL')
    expect(context[0].content.endsWith(tailQuestion)).toBe(true)
    expect(bytes).toBeLessThanOrEqual(MAX_INFERENCE_TEXT_BYTES)
  })

  it('preserves a UTF-8-safe multilingual suffix without splitting its final question', () => {
    const tailQuestion = '💡末尾问题：请比较速度、显存和答案质量。🤖'
    const context = inferenceContext([{
      id: 'oversized-multibyte-latest',
      role: 'user',
      content: `${'较早的多语言上下文🌏'.repeat(4_000)}${tailQuestion}`,
    }], '', 0)
    const content = context[0].content

    expect(new TextEncoder().encode(content).byteLength).toBeLessThanOrEqual(MAX_INFERENCE_TEXT_BYTES)
    expect(content).not.toContain('\uFFFD')
    expect(content.endsWith(tailQuestion)).toBe(true)
    expect(content).toMatch(/^\[The beginning of this turn was omitted to fit/)
  })

  it('follows streaming only while the reader remains near the transcript bottom', () => {
    expect(isTranscriptNearBottom({ scrollHeight: 2_000, scrollTop: 1_250, clientHeight: 700 }))
      .toBe(true)
    expect(isTranscriptNearBottom({ scrollHeight: 2_000, scrollTop: 700, clientHeight: 700 }))
      .toBe(false)
  })

  it('compacts only after a meaningful unsummarized window accumulates', () => {
    expect(shouldAutoCompact(19, 0)).toBe(false)
    expect(shouldAutoCompact(20, 0)).toBe(true)
    expect(shouldAutoCompact(31, 12)).toBe(false)
    expect(shouldAutoCompact(32, 12)).toBe(true)
  })

  it('formats recent database timestamps for the history rail', () => {
    const now = Date.parse('2026-08-09T10:00:00Z')
    expect(formatConversationTime('2026-08-09T09:58:00Z', now)).toBe('2m ago')
    expect(formatConversationTime('not-a-date', now)).toBe('Saved locally')
  })

  it('retries a title-only save against the latest revision after one conflict', async () => {
    const revisions: number[] = []
    const save = async (revision: number) => {
      revisions.push(revision)
      if (revision === 3) throw { status: 409 }
      return { revision: revision + 1, title: 'Renamed' }
    }

    const result = await saveWithRevisionRetry(
      3,
      save,
      async () => ({ revision: 7, title: 'Changed elsewhere' }),
      (reason) => Boolean(reason && typeof reason === 'object' && 'status' in reason && reason.status === 409),
    )

    expect(revisions).toEqual([3, 7])
    expect(result).toEqual({ value: { revision: 8, title: 'Renamed' }, recovered: true })
  })

  it('does not reload or retry a title save for non-conflict failures', async () => {
    const failure = new Error('offline')
    let loads = 0

    await expect(saveWithRevisionRetry(
      2,
      async () => { throw failure },
      async () => { loads += 1; return { revision: 9 } },
      () => false,
    )).rejects.toBe(failure)
    expect(loads).toBe(0)
  })

  it('reloads a stale delete conflict without retrying against the newer revision', async () => {
    const attemptedRevisions: number[] = []
    const latest = { revision: 8, messages: ['newer tab answer'] }

    const result = await deleteWithConflictReload(
      7,
      async (revision) => {
        attemptedRevisions.push(revision)
        throw { status: 409 }
      },
      async () => latest,
      (reason) => Boolean(reason && typeof reason === 'object' && 'status' in reason && reason.status === 409),
    )

    expect(attemptedRevisions).toEqual([7])
    expect(result).toEqual({ deleted: false, latest })
  })
})

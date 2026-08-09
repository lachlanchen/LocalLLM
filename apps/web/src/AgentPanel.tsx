import { Bot, CheckCircle2, ChevronDown, CircleStop, Play, ShieldCheck, TerminalSquare } from 'lucide-react'
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import {
  AgentApiError,
  agentApi,
  boundedAgentOutput,
  type AgentCapabilities,
  type AgentCapability,
  type AgentCodeExecution,
  type AgentPlanProposal,
  type PythonPlanStep,
} from './agentApi'
import './agent-panel.css'

export interface AgentPanelProps {
  goal: string
  model: string
  hasImage?: boolean
  disabled?: boolean
  onAppendResult: (markdown: string) => void
  onBusyChange?: (busy: boolean) => void
}

type Operation = 'capabilities' | 'planning' | `review:${string}` | `run:${string}`

interface ReviewedCode {
  token: string
  expiresAt: string
  codeSha256: string
}

interface OperationHandle {
  controller: AbortController
  generation: number
}

export const MAX_PERSISTED_AGENT_RESULT_CHARS = 30_000
const MAX_STREAM_MARKDOWN_CHARS = 14_000

export interface AgentBusyNotifier {
  readonly active: boolean
  start: () => void
  stop: () => void
}

export function createAgentBusyNotifier(
  getListener: () => ((busy: boolean) => void) | undefined,
): AgentBusyNotifier {
  let active = false
  const emit = (next: boolean): void => {
    try {
      getListener()?.(next)
    } catch {
      // A parent observer cannot break request cleanup or leave the gate locked.
    }
  }
  return {
    get active() { return active },
    start() {
      if (active) return
      active = true
      emit(true)
    },
    stop() {
      if (!active) return
      active = false
      emit(false)
    },
  }
}

function isAbort(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

function friendlyError(error: unknown): string {
  if (error instanceof AgentApiError) return error.message
  if (error instanceof Error && error.message) return error.message.slice(0, 400)
  return 'The local Agent request did not complete.'
}

export function availableAgentCapabilities(
  capabilities: AgentCapabilities | null,
  hasImage: boolean,
): AgentCapability[] {
  const enabled: AgentCapability[] = ['respond', 'web_search', 'paper_search']
  if (hasImage) enabled.push('vision')
  const python = capabilities?.capabilities.find((capability) => capability.id === 'python')
  if (
    capabilities?.operator_code_execution_enabled
    && capabilities.sandbox_ready
    && python?.available
  ) {
    enabled.push('python')
  }
  return enabled
}

function indentedMarkdown(value: string): { markdown: string; truncated: boolean } {
  if (!value) return { markdown: '_No output._', truncated: false }
  let markdown = ''
  let truncated = false
  const lines = value.split('\n')
  for (let index = 0; index < lines.length; index += 1) {
    const prefix = `${markdown ? '\n' : ''}    `
    const available = MAX_STREAM_MARKDOWN_CHARS - markdown.length - prefix.length
    if (available <= 0) {
      truncated = true
      break
    }
    const line = lines[index]
    markdown += prefix + line.slice(0, available)
    if (
      line.length > available
      || (index < lines.length - 1 && markdown.length >= MAX_STREAM_MARKDOWN_CHARS)
    ) {
      truncated = true
      break
    }
  }
  return { markdown, truncated }
}

export function executionResultMarkdown(execution: AgentCodeExecution): string {
  const boundedStdout = boundedAgentOutput(execution.result.stdout)
  const boundedStderr = boundedAgentOutput(execution.result.stderr)
  const stdout = indentedMarkdown(boundedStdout)
  const stderr = indentedMarkdown(boundedStderr)
  const locallyTruncated = (
    boundedStdout.length < execution.result.stdout.length
    || boundedStderr.length < execution.result.stderr.length
    || stdout.truncated
    || stderr.truncated
  )
  const exit = execution.result.exit_code === null ? 'none' : String(execution.result.exit_code)
  const markdown = [
    '### Isolated Python result',
    '',
    `- Status: \`${execution.result.status}\``,
    `- Exit code: \`${exit}\``,
    `- Duration: \`${Math.max(0, execution.result.duration_ms)} ms\``,
    `- Output truncated: \`${execution.result.output_truncated || locallyTruncated ? 'yes' : 'no'}\``,
    '',
    '**stdout**',
    '',
    stdout.markdown,
    '',
    '**stderr**',
    '',
    stderr.markdown,
  ].join('\n')
  return markdown.slice(0, MAX_PERSISTED_AGENT_RESULT_CHARS)
}

export function stagedStepLabel(capability: AgentCapability): string {
  return capability === 'python' ? 'approval required' : 'handled by Auto Send'
}

function capabilityLabel(capability: AgentCapability): string {
  return {
    respond: 'Answer',
    web_search: 'Web',
    paper_search: 'Papers',
    vision: 'Vision',
    python: 'Python',
  }[capability]
}

function PythonStepControls({
  step,
  reviewed,
  result,
  busy,
  disabled,
  onReview,
  onRun,
}: {
  step: PythonPlanStep
  reviewed?: ReviewedCode
  result?: AgentCodeExecution
  busy: boolean
  disabled: boolean
  onReview: () => void
  onRun: () => void
}) {
  const expires = reviewed ? new Date(reviewed.expiresAt) : null
  return (
    <div className="agent-panel__python">
      <div className="agent-panel__code-heading">
        <TerminalSquare size={16} aria-hidden="true" />
        <strong>Proposed isolated Python</strong>
      </div>
      <pre className="agent-panel__code" tabIndex={0} aria-label={`Python proposed for ${step.id}`}>
        <code>{step.arguments.code}</code>
      </pre>
      <div className="agent-panel__safety-note">
        <ShieldCheck size={15} aria-hidden="true" />
        <span>No network, host mounts, or writable root. Reviewing does not execute code.</span>
      </div>
      {!reviewed ? (
        <button
          className="agent-panel__button agent-panel__button--review"
          type="button"
          disabled={busy || disabled}
          onClick={onReview}
        >
          <ShieldCheck size={15} aria-hidden="true" />
          Review isolated Python
        </button>
      ) : (
        <div className="agent-panel__run-row">
          <span className="agent-panel__reviewed">
            <CheckCircle2 size={14} aria-hidden="true" />
            Exact code reviewed{expires && !Number.isNaN(expires.valueOf())
              ? ` · expires ${expires.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`
              : ''}
          </span>
          <button
            className="agent-panel__button agent-panel__button--run"
            type="button"
            disabled={busy || disabled}
            onClick={onRun}
          >
            <Play size={15} aria-hidden="true" />
            Run isolated Python
          </button>
        </div>
      )}
      {result && (
        <div className="agent-panel__execution" aria-label="Isolated Python result">
          <div className="agent-panel__execution-meta">
            <strong data-agent-status={result.result.status}>{result.result.status}</strong>
            <span>{result.result.duration_ms} ms</span>
            <span>exit {result.result.exit_code ?? '—'}</span>
            {result.result.output_truncated && <span>output capped</span>}
          </div>
          <div className="agent-panel__streams">
            <div>
              <small>stdout</small>
              <pre tabIndex={0}>{boundedAgentOutput(result.result.stdout) || 'No stdout.'}</pre>
            </div>
            <div>
              <small>stderr</small>
              <pre tabIndex={0}>{boundedAgentOutput(result.result.stderr) || 'No stderr.'}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export function AgentPanel({
  goal,
  model,
  hasImage = false,
  disabled = false,
  onAppendResult,
  onBusyChange,
}: AgentPanelProps) {
  const contentId = useId()
  const [expanded, setExpanded] = useState(false)
  const [capabilities, setCapabilities] = useState<AgentCapabilities | null>(null)
  const [capabilitiesLoaded, setCapabilitiesLoaded] = useState(false)
  const [proposal, setProposal] = useState<AgentPlanProposal | null>(null)
  const [reviewed, setReviewed] = useState<Record<string, ReviewedCode>>({})
  const [results, setResults] = useState<Record<string, AgentCodeExecution>>({})
  const [busy, setBusy] = useState<Operation | null>(null)
  const [error, setError] = useState<string | null>(null)

  const mountedRef = useRef(true)
  const busyRef = useRef<Operation | null>(null)
  const generationRef = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)
  const appendResultRef = useRef(onAppendResult)
  const busyChangeRef = useRef(onBusyChange)
  const busyNotifierRef = useRef<AgentBusyNotifier | null>(null)
  if (!busyNotifierRef.current) {
    busyNotifierRef.current = createAgentBusyNotifier(() => busyChangeRef.current)
  }
  appendResultRef.current = onAppendResult
  busyChangeRef.current = onBusyChange

  const enabledCapabilities = useMemo(
    () => availableAgentCapabilities(capabilities, hasImage),
    [capabilities, hasImage],
  )

  const beginOperation = useCallback((operation: Operation): OperationHandle | null => {
    if (busyRef.current) return null
    const controller = new AbortController()
    const generation = generationRef.current + 1
    generationRef.current = generation
    controllerRef.current = controller
    busyRef.current = operation
    setBusy(operation)
    busyNotifierRef.current?.start()
    return { controller, generation }
  }, [])

  const isCurrent = useCallback((handle: OperationHandle): boolean => (
    mountedRef.current
    && generationRef.current === handle.generation
    && controllerRef.current === handle.controller
    && !handle.controller.signal.aborted
  ), [])

  const finishOperation = useCallback((handle: OperationHandle): void => {
    if (!isCurrent(handle)) return
    controllerRef.current = null
    busyRef.current = null
    setBusy(null)
    busyNotifierRef.current?.stop()
  }, [isCurrent])

  const cancelActive = useCallback((updateUi = true): void => {
    const active = busyRef.current
    generationRef.current += 1
    controllerRef.current?.abort()
    controllerRef.current = null
    busyRef.current = null
    busyNotifierRef.current?.stop()
    if (updateUi && mountedRef.current) {
      setBusy(null)
      if (active === 'capabilities') setCapabilitiesLoaded(true)
    }
  }, [])

  const loadCapabilities = useCallback(async (): Promise<void> => {
    const handle = beginOperation('capabilities')
    if (!handle) return
    setError(null)
    try {
      const response = await agentApi.capabilities(handle.controller.signal)
      if (!isCurrent(handle)) return
      setCapabilities(response)
      setCapabilitiesLoaded(true)
    } catch (caught) {
      if (!isCurrent(handle) || isAbort(caught)) return
      setCapabilities(null)
      setCapabilitiesLoaded(true)
      setError(`Agent capability check: ${friendlyError(caught)}`)
    } finally {
      finishOperation(handle)
    }
  }, [beginOperation, finishOperation, isCurrent])

  useEffect(() => {
    if (expanded && !disabled && !capabilitiesLoaded && !busyRef.current) {
      void loadCapabilities()
    }
  }, [capabilitiesLoaded, disabled, expanded, loadCapabilities])

  useEffect(() => {
    cancelActive()
    setProposal(null)
    setReviewed({})
    setResults({})
    setError(null)
  }, [cancelActive, goal, hasImage, model])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      cancelActive(false)
    }
  }, [cancelActive])

  useEffect(() => {
    if (!disabled) return
    cancelActive()
    setReviewed({})
  }, [cancelActive, disabled])

  const toggleExpanded = (): void => {
    if (expanded) {
      const wasCheckingCapabilities = busyRef.current === 'capabilities'
      cancelActive()
      setReviewed({})
      if (wasCheckingCapabilities) setCapabilitiesLoaded(false)
    }
    setExpanded((current) => !current)
  }

  const planTask = async (): Promise<void> => {
    if (disabled) return
    const trimmedGoal = goal.trim()
    if (!trimmedGoal) {
      setError('Write a goal before asking Agent mode to plan it.')
      return
    }
    const handle = beginOperation('planning')
    if (!handle) return
    setError(null)
    setProposal(null)
    setReviewed({})
    setResults({})
    try {
      const response = await agentApi.propose(
        trimmedGoal,
        model,
        enabledCapabilities,
        handle.controller.signal,
      )
      if (!isCurrent(handle)) return
      setProposal(response)
    } catch (caught) {
      if (!isCurrent(handle) || isAbort(caught)) return
      setError(`Agent planning: ${friendlyError(caught)}`)
    } finally {
      finishOperation(handle)
    }
  }

  const reviewPython = async (step: PythonPlanStep): Promise<void> => {
    if (disabled) return
    const handle = beginOperation(`review:${step.id}`)
    if (!handle) return
    setError(null)
    try {
      const confirmation = await agentApi.confirmPython(step.arguments.code, handle.controller.signal)
      if (!isCurrent(handle)) return
      setReviewed((current) => ({
        ...current,
        [step.id]: {
          token: confirmation.confirmation_token,
          expiresAt: confirmation.expires_at,
          codeSha256: confirmation.code_sha256,
        },
      }))
    } catch (caught) {
      if (!isCurrent(handle) || isAbort(caught)) return
      setError(`Python review: ${friendlyError(caught)}`)
    } finally {
      finishOperation(handle)
    }
  }

  const runPython = async (step: PythonPlanStep, confirmation: ReviewedCode): Promise<void> => {
    if (disabled) return
    if (Date.parse(confirmation.expiresAt) <= Date.now()) {
      setReviewed((current) => {
        const next = { ...current }
        delete next[step.id]
        return next
      })
      setError('That code review expired. Review the exact Python again before running it.')
      return
    }
    const handle = beginOperation(`run:${step.id}`)
    if (!handle) return
    setError(null)
    setReviewed((current) => {
      const next = { ...current }
      delete next[step.id]
      return next
    })
    try {
      const execution = await agentApi.executePython(
        step.arguments.code,
        step.arguments.timeout_seconds,
        confirmation.token,
        handle.controller.signal,
      )
      if (!isCurrent(handle)) return
      setResults((current) => ({ ...current, [step.id]: execution }))
      try {
        appendResultRef.current(executionResultMarkdown(execution))
      } catch {
        setError('Python finished, but its result could not be appended to the conversation.')
      }
    } catch (caught) {
      if (!isCurrent(handle) || isAbort(caught)) return
      const expired = caught instanceof AgentApiError && caught.status === 409
      setError(expired
        ? 'The single-use code review expired or was consumed. Review the exact Python again.'
        : `Isolated Python: ${friendlyError(caught)}`)
    } finally {
      finishOperation(handle)
    }
  }

  const pythonById = useMemo(() => {
    const entries = proposal?.plan.steps
      .filter((step): step is PythonPlanStep => step.capability === 'python')
      .map((step) => [step.id, step] as const) ?? []
    return new Map(entries)
  }, [proposal])

  const pythonReady = enabledCapabilities.includes('python')
  const capabilitySummary = enabledCapabilities.map(capabilityLabel).join(' · ')

  return (
    <section
      className="agent-panel"
      aria-label="Optional Agent mode"
      aria-disabled={disabled}
      data-expanded={expanded}
    >
      <button
        className="agent-panel__toggle"
        type="button"
        disabled={disabled && !expanded}
        aria-expanded={expanded}
        aria-controls={contentId}
        onClick={toggleExpanded}
      >
        <span className="agent-panel__toggle-icon" aria-hidden="true"><Bot size={17} /></span>
        <span>
          <strong>Agent mode</strong>
          <small>{expanded ? 'Plan first · tools stay gated' : 'Off · open when a task needs tools'}</small>
        </span>
        <ChevronDown className="agent-panel__chevron" size={17} aria-hidden="true" />
      </button>

      {expanded && (
        <div
          id={contentId}
          className="agent-panel__content"
          aria-busy={Boolean(busy)}
        >
          <div className="agent-panel__intro">
            <div>
              <span className="agent-panel__eyebrow">PASSIVE PLANNER</span>
              <p>
                Only the current goal is sent. Web, Papers, and Vision run through normal Auto
                Send; this planner never dispatches them.
              </p>
            </div>
            <span className={`agent-panel__sandbox ${pythonReady ? 'is-ready' : ''}`}>
              <ShieldCheck size={14} aria-hidden="true" />
              {pythonReady ? 'Python sandbox ready' : 'Python stays off'}
            </span>
          </div>

          <dl className="agent-panel__facts">
            <div><dt>Model</dt><dd>{model}</dd></div>
            <div><dt>Capabilities</dt><dd>{capabilitySummary}</dd></div>
            <div><dt>Attachment</dt><dd>{hasImage ? 'Image available' : 'No image shared'}</dd></div>
          </dl>

          <div className="agent-panel__actions">
            <button
              className="agent-panel__button agent-panel__button--plan"
              type="button"
              disabled={disabled || Boolean(busy) || !goal.trim() || !capabilitiesLoaded}
              onClick={() => void planTask()}
            >
              <Bot size={16} aria-hidden="true" />
              {busy === 'planning' ? 'Planning…' : 'Plan task'}
            </button>
            {busy && (
              <button
                className="agent-panel__button agent-panel__button--cancel"
                type="button"
                onClick={() => cancelActive()}
              >
                <CircleStop size={15} aria-hidden="true" />
                Cancel
              </button>
            )}
          </div>

          {busy === 'capabilities' && (
            <p className="agent-panel__status" role="status" aria-live="polite">
              Checking local capabilities…
            </p>
          )}
          {error && <p className="agent-panel__error" role="alert">{error}</p>}

          {proposal && (
            <div className="agent-panel__plan">
              <div className="agent-panel__plan-heading">
                <div>
                  <span className="agent-panel__eyebrow">STAGED · NOT EXECUTED</span>
                  <h3>{proposal.plan.goal}</h3>
                </div>
                <span className={`agent-panel__planner is-${proposal.planner}`}>
                  {proposal.planner === 'local-model' ? 'Local model plan' : 'Safe fallback'}
                </span>
              </div>
              {proposal.warning && (
                <p className="agent-panel__warning" role="status">{proposal.warning}</p>
              )}
              <p className="agent-panel__handoff">
                This is a preview. Use normal Auto Send for Answer, Web, Papers, and Vision.
                Python alone has a separate reviewed run control below.
              </p>
              <ol className="agent-panel__steps">
                {proposal.steps.map((step, index) => {
                  const pythonStep = pythonById.get(step.id)
                  return (
                    <li className="agent-panel__step" key={step.id}>
                      <span className="agent-panel__step-number" aria-hidden="true">
                        {String(index + 1).padStart(2, '0')}
                      </span>
                      <div className="agent-panel__step-body">
                        <div className="agent-panel__step-title">
                          <strong>{capabilityLabel(step.capability)}</strong>
                          <span>{stagedStepLabel(step.capability)}</span>
                        </div>
                        <p>{step.objective}</p>
                        {step.depends_on.length > 0 && (
                          <small>After {step.depends_on.join(', ')}</small>
                        )}
                        {pythonStep && (
                          <PythonStepControls
                            step={pythonStep}
                            reviewed={reviewed[step.id]}
                            result={results[step.id]}
                            busy={Boolean(busy)}
                            disabled={disabled}
                            onReview={() => void reviewPython(pythonStep)}
                            onRun={() => {
                              const confirmation = reviewed[step.id]
                              if (confirmation) void runPython(pythonStep, confirmation)
                            }}
                          />
                        )}
                      </div>
                    </li>
                  )
                })}
              </ol>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

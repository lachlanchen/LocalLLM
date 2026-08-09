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
import { MAX_AGENT_GOAL_CHARS } from './agentRouting'
import './agent-panel.css'

export interface AgentPanelProps {
  goal: string
  model: string
  hasImage?: boolean
  disabled?: boolean
  contextKey: string
  routingEnabled: boolean
  autoRequest?: AgentAutoPlanRequest | null
  onRoutingEnabledChange: (enabled: boolean) => void
  onAutoRequestEnd: (requestId: number, outcome: AgentAutoRequestOutcome) => void
  onAppendResult: (markdown: string, context: AgentResultContext) => Promise<void> | void
  onBusyChange?: (busy: boolean) => void
}

export interface AgentAutoPlanRequest {
  id: number
  goal: string
  model: string
  contextKey: string
}

export type AgentAutoRequestOutcome = 'completed' | 'cancelled' | 'failed' | 'unavailable'

export interface AgentResultContext {
  goal: string
  requestId?: number
}

export interface AgentAutoPlanClaim {
  handledRequestId: number | null
}

export function claimAgentAutoPlan(
  claim: AgentAutoPlanClaim,
  request: AgentAutoPlanRequest,
  contextKey: string,
  blocked: boolean,
  start: () => boolean,
): boolean {
  if (
    blocked
    || request.contextKey !== contextKey
    || claim.handledRequestId === request.id
  ) return false
  if (!start()) return false
  claim.handledRequestId = request.id
  return true
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

type PlannedTaskContext = AgentResultContext

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

export function hasInitiallyRunnablePython(proposal: AgentPlanProposal): boolean {
  return proposal.plan.steps.some((step) => (
    step.capability === 'python' && step.depends_on.length === 0
  ))
}

export function isSingleRunnablePythonPlan(proposal: AgentPlanProposal): boolean {
  const pythonSteps = proposal.plan.steps.filter((step) => step.capability === 'python')
  return pythonSteps.length === 1 && pythonSteps[0].depends_on.length === 0
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
      {!result && (!reviewed ? (
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
      ))}
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
  contextKey,
  routingEnabled,
  autoRequest = null,
  onRoutingEnabledChange,
  onAutoRequestEnd,
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
  const autoRequestEndRef = useRef(onAutoRequestEnd)
  const plannedTaskRef = useRef<PlannedTaskContext | null>(null)
  const autoPlanClaimRef = useRef<AgentAutoPlanClaim>({ handledRequestId: null })
  const contextKeyRef = useRef(contextKey)
  const busyNotifierRef = useRef<AgentBusyNotifier | null>(null)
  if (!busyNotifierRef.current) {
    busyNotifierRef.current = createAgentBusyNotifier(() => busyChangeRef.current)
  }
  appendResultRef.current = onAppendResult
  busyChangeRef.current = onBusyChange
  autoRequestEndRef.current = onAutoRequestEnd

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

  const endAutoRequest = useCallback((
    outcome: AgentAutoRequestOutcome,
    explicitRequestId?: number,
  ): void => {
    const requestId = explicitRequestId ?? plannedTaskRef.current?.requestId
    plannedTaskRef.current = null
    if (requestId === undefined) return
    try {
      autoRequestEndRef.current(requestId, outcome)
    } catch {
      // Parent cleanup is advisory; the panel must always release its own lane.
    }
  }, [])

  const clearPlan = useCallback((
    outcome?: AgentAutoRequestOutcome,
    explicitRequestId?: number,
  ): void => {
    cancelActive()
    setProposal(null)
    setReviewed({})
    setResults({})
    setError(null)
    if (outcome) endAutoRequest(outcome, explicitRequestId)
    else plannedTaskRef.current = null
  }, [cancelActive, endAutoRequest])

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
    if (expanded && !autoRequest && !disabled && !capabilitiesLoaded && !busyRef.current) {
      void loadCapabilities()
    }
  }, [autoRequest, capabilitiesLoaded, disabled, expanded, loadCapabilities])

  useEffect(() => {
    if (autoRequest) return
    clearPlan()
  }, [autoRequest, clearPlan, goal, hasImage, model])

  useEffect(() => {
    const previous = contextKeyRef.current
    contextKeyRef.current = contextKey
    if (previous === contextKey) return
    // The auto request is created from the newly persisted conversation, so
    // that same transition is its binding rather than an invalidation.
    if (autoRequest?.contextKey === contextKey) return
    clearPlan(autoRequest ? 'cancelled' : undefined, autoRequest?.id)
  }, [autoRequest, clearPlan, contextKey])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      cancelActive(false)
    }
  }, [cancelActive])

  useEffect(() => {
    if (!disabled) return
    clearPlan(autoRequest ? 'cancelled' : undefined, autoRequest?.id)
  }, [autoRequest, clearPlan, disabled])

  const toggleExpanded = (): void => {
    if (expanded) {
      const wasCheckingCapabilities = busyRef.current === 'capabilities'
      cancelActive()
      setReviewed({})
      if (wasCheckingCapabilities) setCapabilitiesLoaded(false)
    }
    setExpanded((current) => !current)
  }

  const planTask = useCallback((request?: AgentAutoPlanRequest): boolean => {
    if (disabled) return false
    const trimmedGoal = (request?.goal ?? goal).trim()
    if (!trimmedGoal) {
      setError('Write a goal before asking Agent mode to plan it.')
      if (request) {
        plannedTaskRef.current = { goal: trimmedGoal, requestId: request.id }
        endAutoRequest('failed')
      }
      return false
    }
    if (trimmedGoal.length > MAX_AGENT_GOAL_CHARS) {
      setExpanded(true)
      setError(`Agent goals are limited to ${MAX_AGENT_GOAL_CHARS.toLocaleString()} characters. Shorten this execution request and try again.`)
      if (request) {
        plannedTaskRef.current = { goal: trimmedGoal, requestId: request.id }
        endAutoRequest('failed')
      }
      return false
    }
    const handle = beginOperation('planning')
    if (!handle) return false
    const requestedModel = request?.model ?? model
    plannedTaskRef.current = {
      goal: trimmedGoal,
      ...(request ? { requestId: request.id } : {}),
    }
    setExpanded(true)
    setError(null)
    setProposal(null)
    setReviewed({})
    setResults({})
    void (async () => {
      try {
        let resolvedCapabilities = capabilities
        if (!capabilitiesLoaded || !resolvedCapabilities) {
          resolvedCapabilities = await agentApi.capabilities(handle.controller.signal)
          if (!isCurrent(handle)) return
          setCapabilities(resolvedCapabilities)
          setCapabilitiesLoaded(true)
        }
        const available = availableAgentCapabilities(resolvedCapabilities, hasImage)
        const proposalCapabilities: AgentCapability[] = request
          ? available.includes('python') ? ['respond', 'python'] : ['respond']
          : available
        if (request && !proposalCapabilities.includes('python')) {
          setError('Isolated Python is unavailable. Nothing ran; enable the operator sandbox and verify its local runtime before retrying.')
          endAutoRequest('unavailable')
          return
        }
        const response = await agentApi.propose(
          trimmedGoal,
          requestedModel,
          proposalCapabilities,
          handle.controller.signal,
        )
        if (!isCurrent(handle)) return
        setProposal(response)
        if (request && !isSingleRunnablePythonPlan(response)) {
          setError('Agent did not stage exactly one runnable isolated-Python step. Nothing ran; ask for one self-contained script, or turn Agent routing off for a normal answer.')
          endAutoRequest('unavailable')
        }
      } catch (caught) {
        if (!isCurrent(handle) || isAbort(caught)) return
        setCapabilitiesLoaded(true)
        setError(`Agent planning: ${friendlyError(caught)} Nothing ran.`)
        if (request) endAutoRequest('failed')
      } finally {
        finishOperation(handle)
      }
    })()
    return true
  }, [
    beginOperation,
    capabilities,
    capabilitiesLoaded,
    disabled,
    endAutoRequest,
    finishOperation,
    goal,
    hasImage,
    isCurrent,
    model,
  ])

  useEffect(() => {
    if (!autoRequest) return
    claimAgentAutoPlan(
      autoPlanClaimRef.current,
      autoRequest,
      contextKey,
      Boolean(disabled || busy),
      () => planTask(autoRequest),
    )
  }, [autoRequest, busy, contextKey, disabled, planTask])

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
      const taskContext = plannedTaskRef.current
      try {
        if (!taskContext) throw new Error('The staged task context is no longer available.')
        await appendResultRef.current(executionResultMarkdown(execution), {
          goal: taskContext.goal,
          ...(taskContext.requestId === undefined ? {} : { requestId: taskContext.requestId }),
        })
        if (taskContext.requestId !== undefined) endAutoRequest('completed')
      } catch (caught) {
        setError(caught instanceof Error && caught.message
          ? `Python finished, but its result could not be saved: ${caught.message.slice(0, 300)}`
          : 'Python finished, but its result could not be appended to the conversation.')
        // Keep the completed output visible and the task bound until the user
        // explicitly discards it; a consumed execution cannot be retried safely.
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

  const discardPlan = (): void => {
    clearPlan(plannedTaskRef.current?.requestId === undefined ? undefined : 'cancelled')
  }

  const changeRouting = (): void => {
    if (busy) return
    const next = !routingEnabled
    if (!next) discardPlan()
    try {
      onRoutingEnabledChange(next)
    } catch {
      setError('The Agent routing preference could not be updated.')
    }
  }

  return (
    <section
      className="agent-panel"
      aria-label="Optional Agent mode"
      aria-disabled={disabled}
      data-expanded={expanded}
      data-routing-enabled={routingEnabled}
    >
      <div className="agent-panel__header">
        <button
          className="agent-panel__toggle"
          type="button"
          disabled={disabled || Boolean(busy)}
          aria-label={expanded ? 'Close Agent details' : 'Open Agent details'}
          aria-expanded={expanded}
          aria-controls={contentId}
          onClick={toggleExpanded}
        >
          <span className="agent-panel__toggle-icon" aria-hidden="true"><Bot size={17} /></span>
          <span>
            <strong>Agent mode</strong>
            <small>{expanded
              ? 'Plan first · exact code stays gated'
              : routingEnabled
                ? 'On · execution requests open a reviewed plan'
                : 'Off · Send answers normally'}</small>
          </span>
          <ChevronDown className="agent-panel__chevron" size={17} aria-hidden="true" />
        </button>
        <button
          className={`agent-panel__switch ${routingEnabled ? 'is-on' : ''}`}
          type="button"
          role="switch"
          aria-checked={routingEnabled}
          aria-label="Route explicit Python execution requests to Agent mode"
          data-testid="agent-routing-toggle"
          disabled={disabled || Boolean(busy)}
          onClick={changeRouting}
        >
          <span aria-hidden="true"><i /></span>
          <strong>{routingEnabled ? 'On' : 'Off'}</strong>
        </button>
      </div>

      {expanded && (
        <div
          id={contentId}
          className="agent-panel__content"
          aria-busy={Boolean(busy)}
        >
          <div className="agent-panel__intro">
            <div>
              <span className="agent-panel__eyebrow">REVIEWED AGENT PLANNER</span>
              <p>
                Agent routing catches explicit Python execution requests. It saves the request,
                stages exact code, and waits for your separate Review and Run actions.
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
              disabled={disabled || Boolean(busy) || !goal.trim()}
              onClick={() => void planTask()}
            >
              <Bot size={16} aria-hidden="true" />
              {busy === 'planning' ? 'Planning…' : 'Plan task'}
            </button>
            {busy && (
              <button
                className="agent-panel__button agent-panel__button--cancel"
                type="button"
                onClick={() => clearPlan(plannedTaskRef.current?.requestId === undefined ? undefined : 'cancelled')}
              >
                <CircleStop size={15} aria-hidden="true" />
                Cancel
              </button>
            )}
            {!busy && proposal && (
              <button
                className="agent-panel__button agent-panel__button--cancel"
                type="button"
                onClick={discardPlan}
              >
                Discard plan
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
                This is a preview, not an execution. Python runs only after you review the exact
                displayed code and press Run isolated Python.
              </p>
              <ol className="agent-panel__steps">
                {proposal.steps.map((step, index) => {
                  const pythonStep = pythonById.get(step.id)
                  const pythonDependenciesReady = pythonStep?.depends_on.every((dependencyId) => (
                    pythonById.has(dependencyId)
                    && results[dependencyId]?.result.status === 'succeeded'
                  )) ?? false
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
                            disabled={disabled || !pythonDependenciesReady}
                            onReview={() => void reviewPython(pythonStep)}
                            onRun={() => {
                              const confirmation = reviewed[step.id]
                              if (confirmation) void runPython(pythonStep, confirmation)
                            }}
                          />
                        )}
                        {pythonStep && !pythonDependenciesReady && (
                          <small>Run controls stay locked until every executable Python dependency succeeds.</small>
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

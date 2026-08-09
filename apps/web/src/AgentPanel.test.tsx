import { StrictMode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import {
  AgentPanel,
  availableAgentCapabilities,
  createAgentBusyNotifier,
  executionResultMarkdown,
  MAX_PERSISTED_AGENT_RESULT_CHARS,
  stagedStepLabel,
} from './AgentPanel'
import type { AgentCapabilities, AgentCodeExecution } from './agentApi'

function capabilities(pythonReady: boolean): AgentCapabilities {
  return {
    schema_version: '1',
    default_mode: 'ordinary_chat',
    ordinary_chat_auto_executes_tools: false,
    operator_code_execution_enabled: pythonReady,
    capabilities: [
      {
        id: 'plan_validation',
        available: true,
        default_enabled: false,
        invocation: 'explicit_endpoint',
      },
      {
        id: 'python',
        available: pythonReady,
        default_enabled: false,
        invocation: 'two_step_confirmation',
      },
    ],
    sandbox_image: 'localllm/python-sandbox:fixed',
    sandbox_profile: 'python-v1',
    sandbox_ready: pythonReady,
    limits: {
      network: 'none',
      host_mounts: false,
      root_filesystem: 'read_only',
      user: '65532:65532',
      capabilities: 'dropped',
      no_new_privileges: true,
      workdir: 'ephemeral_tmpfs',
      pids: 64,
      memory_mib: 512,
      cpus: 1,
      max_output_bytes: 65536,
      max_seconds: 20,
      max_parallel: 2,
    },
  }
}

describe('Agent panel', () => {
  it('is reusable, disabled-aware, and collapsed by default under StrictMode', () => {
    const html = renderToStaticMarkup(
      <StrictMode>
        <AgentPanel
          goal="Plan this task"
          model="qwen3:8b-q4_K_M"
          disabled
          onAppendResult={vi.fn()}
        />
      </StrictMode>,
    )

    expect(html).toContain('aria-label="Optional Agent mode"')
    expect(html).toContain('aria-disabled="true"')
    expect(html).toContain('aria-expanded="false"')
    expect(html).toContain('Off · open when a task needs tools')
    expect(html).not.toContain('Plan task')
    expect(html).not.toContain('current goal is sent')
  })

  it('enables vision only for an attachment and Python only for operator-ready sandbox', () => {
    expect(availableAgentCapabilities(null, false)).toEqual([
      'respond',
      'web_search',
      'paper_search',
    ])
    expect(availableAgentCapabilities(capabilities(false), true)).toEqual([
      'respond',
      'web_search',
      'paper_search',
      'vision',
    ])
    expect(availableAgentCapabilities(capabilities(true), true)).toEqual([
      'respond',
      'web_search',
      'paper_search',
      'vision',
      'python',
    ])
  })

  it('makes the normal Auto Send handoff distinct from explicit Python approval', () => {
    expect(stagedStepLabel('web_search')).toBe('handled by Auto Send')
    expect(stagedStepLabel('paper_search')).toBe('handled by Auto Send')
    expect(stagedStepLabel('vision')).toBe('handled by Auto Send')
    expect(stagedStepLabel('python')).toBe('approval required')
  })

  it('releases the parent busy signal on abort or unmount cleanup without sticking', () => {
    const changes: boolean[] = []
    const notifier = createAgentBusyNotifier(() => (value) => changes.push(value))

    notifier.start()
    expect(notifier.active).toBe(true)
    notifier.stop()
    notifier.stop()

    expect(notifier.active).toBe(false)
    expect(changes).toEqual([true, false])
  })

  it('formats bounded execution output as inert indented Markdown', () => {
    const execution: AgentCodeExecution = {
      execution_id: `exec_${'a'.repeat(32)}`,
      events: [],
      result: {
        status: 'failed',
        exit_code: 1,
        stdout: '# not a heading\n```not a fence```',
        stderr: 'example failure',
        output_truncated: false,
        duration_ms: 14,
      },
    }

    const markdown = executionResultMarkdown(execution)
    expect(markdown).toContain('- Status: `failed`')
    expect(markdown).toContain('    # not a heading')
    expect(markdown).toContain('    ```not a fence```')
    expect(markdown).toContain('    example failure')
  })

  it('keeps appended execution output within the durable conversation limit', () => {
    const execution: AgentCodeExecution = {
      execution_id: `exec_${'b'.repeat(32)}`,
      events: [],
      result: {
        status: 'succeeded',
        exit_code: 0,
        stdout: `${'x\n'.repeat(40_000)}stdout tail`,
        stderr: `${'y\n'.repeat(40_000)}stderr tail`,
        output_truncated: false,
        duration_ms: 20,
      },
    }

    const markdown = executionResultMarkdown(execution)
    expect(markdown.length).toBeLessThanOrEqual(MAX_PERSISTED_AGENT_RESULT_CHARS)
    expect(markdown.length).toBeLessThan(32_000)
    expect(markdown).toContain('- Output truncated: `yes`')
  })
})

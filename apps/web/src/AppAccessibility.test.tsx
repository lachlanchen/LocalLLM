import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import App from './App'

afterEach(() => vi.unstubAllGlobals())

describe('application accessibility landmarks', () => {
  it('labels navigation and composer controls without relying on icon or placeholder text', () => {
    const html = renderToStaticMarkup(<App />)

    expect(html).toContain('aria-label="Open navigation"')
    expect(html).toContain('aria-controls="primary-sidebar"')
    expect(html).toContain('aria-label="LocalLLM navigation"')
    expect(html).toContain('aria-label="Primary workspace"')
    expect(html).toContain('aria-current="page"')
    expect(html).toContain('aria-label="Message LocalLLM"')
    expect(html).toContain('aria-describedby="composer-privacy-note"')
    expect(html).toContain('maxLength="32000"')
    expect(html).toContain('aria-label="Chat composer"')
  })

  it('exposes the initially open desktop history rail as a named session list', () => {
    vi.stubGlobal('window', { innerWidth: 1200 })

    const html = renderToStaticMarkup(<App />)

    expect(html).toContain('aria-label="Close saved conversations"')
    expect(html).toContain('aria-controls="chat-history-panel"')
    expect(html).toContain('aria-label="Saved conversations"')
    expect(html).toContain('role="list"')
    expect(html).toContain('aria-label="Close conversation history"')
  })

  it('keeps the mounted chat transcript in a dedicated keyboard-scrollable region', () => {
    const html = renderToStaticMarkup(<App />)

    expect(html).toContain('class="workspace-view workspace-view--chat"')
    expect(html).toContain('data-testid="chat-transcript"')
    expect(html).toContain('aria-label="Conversation transcript"')
    expect(html).toContain('tabindex="0"')
  })
})

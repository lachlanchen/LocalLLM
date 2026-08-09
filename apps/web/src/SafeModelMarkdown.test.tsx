import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { SafeModelMarkdown } from './SafeModelMarkdown'

function render(markdown: string): string {
  return renderToStaticMarkup(<SafeModelMarkdown>{markdown}</SafeModelMarkdown>)
}

describe('safe model-authored Markdown', () => {
  it('renders inline, reference, automatic, and literal links as inert text', () => {
    const html = render([
      '[inline label](https://attacker.example/inline)',
      '[reference label][target]',
      '<https://attacker.example/autolink>',
      'https://attacker.example/literal',
      '',
      '[target]: https://attacker.example/reference',
    ].join('\n'))

    expect(html).not.toContain('<a')
    expect(html).not.toContain('href=')
    expect(html).toContain('inline label')
    expect(html).toContain('reference label')
    expect(html).toContain('https://attacker.example/autolink')
    expect(html).toContain('https://attacker.example/literal')
  })

  it('never creates an image request and retains only useful alt text', () => {
    const html = render([
      '![tracking pixel](https://attacker.example/pixel.png)',
      '![data payload](data:image/svg+xml;base64,PHN2Zy8+)',
      '[![nested image](https://attacker.example/nested.webp)](https://attacker.example/click)',
    ].join('\n'))

    expect(html).not.toContain('<img')
    expect(html).not.toContain('src=')
    expect(html).not.toContain('<a')
    expect(html).toContain('tracking pixel')
    expect(html).toContain('data payload')
    expect(html).toContain('nested image')
  })

  it('drops model-authored raw HTML while preserving ordinary formatting', () => {
    const html = render([
      '<a href="https://attacker.example/raw">raw link</a>',
      '<img src="https://attacker.example/raw.png" alt="raw image">',
      '',
      '**Bold evidence** and `code`.',
    ].join('\n'))

    expect(html).not.toContain('attacker.example')
    expect(html).not.toContain('<a')
    expect(html).not.toContain('<img')
    expect(html).toContain('<strong>Bold evidence</strong>')
    expect(html).toContain('<code>code</code>')
  })
})

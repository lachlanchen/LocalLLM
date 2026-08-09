import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { SafeModelMarkdown } from './SafeModelMarkdown'

function render(markdown: string): string {
  return renderToStaticMarkup(<SafeModelMarkdown>{markdown}</SafeModelMarkdown>)
}

describe('safe model-authored Markdown', () => {
  it('renders headings, lists, blockquotes, code, and GFM text features', () => {
    const html = render([
      '# Heading',
      '',
      '> Quoted **evidence**',
      '',
      '- nested item',
      '  1. ordered item',
      '- [x] verified',
      '- [ ] pending',
      '',
      '~~superseded~~ and `inlineCode()`',
      '',
      '```ts',
      'const answer: number = 42',
      '```',
    ].join('\n'))

    expect(html).toContain('<h1>Heading</h1>')
    expect(html).toContain('<blockquote>')
    expect(html).toContain('<strong>evidence</strong>')
    expect(html).toContain('<ul class="contains-task-list">')
    expect(html).toContain('type="checkbox"')
    expect(html).toContain('checked=""')
    expect(html).toContain('<del>superseded</del>')
    expect(html).toContain('<code class="language-ts">')
  })

  it('renders GFM tables inside a keyboard-scrollable region', () => {
    const html = render([
      '| Model | Quant | Ready |',
      '| :---- | ----: | :---: |',
      '| Qwen | Q4 | yes |',
      '| Gemma | Q8 | no |',
    ].join('\n'))

    expect(html).toContain('class="model-markdown-table-scroll"')
    expect(html).toContain('aria-label="Markdown table"')
    expect(html).toContain('tabindex="0"')
    expect(html).toContain('<table>')
    expect(html).toContain('<th style="text-align:left">Model</th>')
    expect(html).toContain('<th style="text-align:right">Quant</th>')
    expect(html).toContain('<th style="text-align:center">Ready</th>')
    expect(html).toContain('<td style="text-align:left">Qwen</td>')
  })

  it('renders inline and display KaTeX from dollar delimiters', () => {
    const html = render([
      'Einstein wrote $E = mc^2$.',
      '',
      '$$',
      '\\int_0^1 x^2 \\, dx = \\frac{1}{3}',
      '$$',
    ].join('\n'))

    expect(html).toContain('class="katex"')
    expect(html).toContain('class="katex-display"')
    expect(html).toContain('<math xmlns="http://www.w3.org/1998/Math/MathML"')
    expect(html).toContain('<annotation encoding="application/x-tex">E = mc^2</annotation>')
  })

  it('supports common LaTeX delimiters but leaves code examples literal', () => {
    const html = render([
      'Inline \\(a^2 + b^2 = c^2\\).',
      '',
      '\\[',
      '\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}',
      '\\]',
      '',
      '`\\(not math\\)`',
      '',
      '```text',
      '\\[also not math\\]',
      '```',
      '',
      '    \\(indented code is literal\\)',
    ].join('\n'))

    expect(html.match(/class="katex"/g)?.length).toBe(2)
    expect(html).toContain('class="katex-display"')
    expect(html).toContain('<code>\\(not math\\)</code>')
    expect(html).toContain('\\[also not math\\]')
    expect(html).toContain('\\(indented code is literal\\)')
  })

  it('uses CommonMark fence boundaries inside blockquotes and lists', () => {
    const html = render([
      '> ~~~text',
      '> \\(quoted literal\\)',
      '> ~~~not-a-closing-fence',
      '> \\[still quoted literal\\]',
      '> ~~~',
      '',
      '- code sample:',
      '  ```text',
      '  \\(listed literal\\)',
      '  ```',
      '',
      'Outside \\(rendered math\\).',
    ].join('\n'))

    expect(html.match(/class="katex"/g)?.length).toBe(1)
    expect(html).toContain('\\(quoted literal\\)')
    expect(html).toContain('\\[still quoted literal\\]')
    expect(html).toContain('\\(listed literal\\)')
    expect(html).toContain('<annotation encoding="application/x-tex">rendered math</annotation>')
  })

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

  it('does not restore navigation or remote images through KaTeX commands', () => {
    const html = render([
      '$\\href{https://attacker.example/math}{leave}$',
      '',
      '$\\includegraphics{https://attacker.example/pixel.png}$',
    ].join('\n'))

    expect(html).not.toMatch(/<a(?:\s|>)/)
    expect(html).not.toMatch(/\shref=/)
    expect(html).not.toMatch(/<img(?:\s|>)/)
    expect(html).not.toMatch(/\ssrc=/)
  })
})

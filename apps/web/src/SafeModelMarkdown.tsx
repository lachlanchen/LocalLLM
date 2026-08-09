import rehypeKatex from 'rehype-katex'
import type { Components } from 'react-markdown'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import 'katex/dist/katex.min.css'

function isEscaped(text: string, index: number): boolean {
  let slashCount = 0

  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) {
    slashCount += 1
  }

  return slashCount % 2 === 1
}

function findUnescaped(text: string, token: string, from: number): number {
  let index = text.indexOf(token, from)

  while (index !== -1 && isEscaped(text, index)) {
    index = text.indexOf(token, index + token.length)
  }

  return index
}

function normalizePlainLatexDelimiters(text: string): string {
  let cursor = 0
  let output = ''

  while (cursor < text.length) {
    const inlineStart = findUnescaped(text, '\\(', cursor)
    const displayStart = findUnescaped(text, '\\[', cursor)
    const candidates = [inlineStart, displayStart].filter((index) => index !== -1)

    if (candidates.length === 0) {
      output += text.slice(cursor)
      break
    }

    const start = Math.min(...candidates)
    const isDisplay = start === displayStart
    const closingToken = isDisplay ? '\\]' : '\\)'
    const end = findUnescaped(text, closingToken, start + 2)

    if (end === -1 || (!isDisplay && text.slice(start + 2, end).includes('\n'))) {
      output += text.slice(cursor, start + 2)
      cursor = start + 2
      continue
    }

    const marker = isDisplay ? '$$' : '$'
    output += text.slice(cursor, start)
    output += marker
    output += text.slice(start + 2, end)
    output += marker
    cursor = end + 2
  }

  return output
}

interface PositionedMarkdownNode {
  type: string
  position?: {
    start: { offset?: number }
    end: { offset?: number }
  }
  children?: PositionedMarkdownNode[]
}

interface SourceRange {
  start: number
  end: number
}

const SOURCE_MARKDOWN_PARSER = unified().use(remarkParse).use(remarkGfm)

function protectedSourceRanges(markdown: string): SourceRange[] {
  const root = SOURCE_MARKDOWN_PARSER.parse(markdown) as unknown as PositionedMarkdownNode
  const ranges: SourceRange[] = []

  const visit = (node: PositionedMarkdownNode): void => {
    if (node.type === 'code' || node.type === 'inlineCode' || node.type === 'html') {
      const start = node.position?.start.offset
      const end = node.position?.end.offset
      if (typeof start === 'number' && typeof end === 'number' && start < end) {
        ranges.push({ start, end })
      }
      return
    }
    node.children?.forEach(visit)
  }
  visit(root)
  ranges.sort((left, right) => left.start - right.start || left.end - right.end)

  const merged: SourceRange[] = []
  for (const range of ranges) {
    const prior = merged.at(-1)
    if (prior && range.start <= prior.end) {
      prior.end = Math.max(prior.end, range.end)
    } else {
      merged.push({ ...range })
    }
  }
  return merged
}

/**
 * Local models frequently use LaTeX's `\\(...\\)` and `\\[...\\]`
 * delimiters instead of Markdown's dollar delimiters. Normalize those forms
 * without touching fenced or inline code, where they must remain literal.
 */
export function normalizeLatexMathDelimiters(markdown: string): string {
  let ranges: SourceRange[]
  try {
    ranges = protectedSourceRanges(markdown)
  } catch {
    // Fail closed for fidelity: malformed Markdown remains literal rather than
    // risking a math rewrite inside source code or raw HTML.
    return markdown
  }
  let cursor = 0
  let output = ''
  for (const range of ranges) {
    output += normalizePlainLatexDelimiters(markdown.slice(cursor, range.start))
    output += markdown.slice(range.start, range.end)
    cursor = range.end
  }
  output += normalizePlainLatexDelimiters(markdown.slice(cursor))
  return output
}

const SAFE_MODEL_COMPONENTS: Components = {
  a: ({ children }) => <span className="model-markdown-link-text">{children}</span>,
  img: ({ alt }) => (
    <span className="model-markdown-image-alt">
      {alt ? `[Image omitted: ${alt}]` : '[Image omitted]'}
    </span>
  ),
  table: ({ children }) => (
    <div className="model-markdown-table-scroll" role="region" aria-label="Markdown table" tabIndex={0}>
      <table>{children}</table>
    </div>
  ),
}

export function SafeModelMarkdown({ children }: { children: string }) {
  return (
    <div className="model-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { trust: false, strict: 'ignore', output: 'htmlAndMathml' }]]}
        components={SAFE_MODEL_COMPONENTS}
        skipHtml
      >
        {normalizeLatexMathDelimiters(children)}
      </ReactMarkdown>
    </div>
  )
}

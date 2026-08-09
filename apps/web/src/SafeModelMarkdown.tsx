import type { Components } from 'react-markdown'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const SAFE_MODEL_COMPONENTS: Components = {
  a: ({ children }) => <span className="model-markdown-link-text">{children}</span>,
  img: ({ alt }) => (
    <span className="model-markdown-image-alt">
      {alt ? `[Image omitted: ${alt}]` : '[Image omitted]'}
    </span>
  ),
}

export function SafeModelMarkdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={SAFE_MODEL_COMPONENTS}
      skipHtml
    >
      {children}
    </ReactMarkdown>
  )
}

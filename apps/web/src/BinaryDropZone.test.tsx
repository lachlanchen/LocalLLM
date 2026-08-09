import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { BinaryDropZone } from './BinaryDropZone'

describe('binary upload drop zone', () => {
  it('disables file selection and advertises its busy state during an operation', () => {
    const html = renderToStaticMarkup(
      <BinaryDropZone busy disabled onFile={vi.fn()} />,
    )

    expect(html).toContain('aria-disabled="true"')
    expect(html).toContain('aria-busy="true"')
    expect(html).toContain('data-status="uploading"')
    expect(html).toContain('disabled=""')
    expect(html).toContain('Inspecting metadata')
  })

  it('keeps an idle drop zone available for the first artifact only', () => {
    const html = renderToStaticMarkup(
      <BinaryDropZone busy={false} disabled={false} onFile={vi.fn()} />,
    )

    expect(html).toContain('aria-disabled="false"')
    expect(html).toContain('data-status="ready"')
    expect(html).not.toContain('disabled=""')
    expect(html).toContain('Choose or drop a binary')
  })
})

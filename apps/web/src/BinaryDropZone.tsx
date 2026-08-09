import { LoaderCircle, UploadCloud } from 'lucide-react'
import type { DragEvent } from 'react'

export function BinaryDropZone({
  busy,
  disabled,
  onFile,
}: {
  busy: boolean
  disabled: boolean
  onFile: (file?: File) => void | Promise<void>
}) {
  const inspectDroppedFile = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    if (disabled) return
    void onFile(event.dataTransfer.files[0])
  }

  return (
    <label
      className={`binary-drop ${disabled ? 'is-disabled' : ''}`}
      aria-disabled={disabled}
      aria-busy={busy}
      data-status={busy ? 'uploading' : disabled ? 'disabled' : 'ready'}
      onDragOver={(event) => event.preventDefault()}
      onDrop={inspectDroppedFile}
    >
      <input
        hidden
        type="file"
        disabled={disabled}
        onChange={(event) => {
          const input = event.currentTarget
          if (disabled) { input.value = ''; return }
          void Promise.resolve(onFile(input.files?.[0])).finally(() => { input.value = '' })
        }}
      />
      {busy ? <LoaderCircle className="spin" size={28} /> : <UploadCloud size={28} />}
      <strong>{busy ? 'Inspecting metadata…' : 'Choose or drop a binary to inspect'}</strong>
      <p>Up to 64 MB · stored locally · static inspection only</p>
    </label>
  )
}

export const MAX_IMAGE_UPLOAD_BYTES = 8 * 1024 * 1024
export const MAX_IMAGE_UPLOADS = 4
export const MAX_TOTAL_IMAGE_UPLOAD_BYTES = 16 * 1024 * 1024
export const MAX_IMAGE_SOURCE_BYTES = 24 * 1024 * 1024

const SUPPORTED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp'])
const HEIF_IMAGE_TYPES = new Set(['image/heic', 'image/heif'])
const HEIF_EXTENSIONS = new Set(['heic', 'heif'])
const HEIF_STILL_BRANDS = new Set(['heic', 'heix', 'heim', 'heis', 'mif1', 'mif2'])
const HEVC_STILL_BRANDS = new Set(['heic', 'heix', 'heim', 'heis'])
const HEIF_SEQUENCE_BRANDS = new Set(['hevc', 'hevx', 'hevm', 'hevs', 'msf1'])
const AVIF_BRANDS = new Set(['avif', 'avis', 'ma1a', 'ma1b'])
const IMAGE_EXTENSION_TYPES: Record<string, string> = {
  jpeg: 'image/jpeg',
  jpg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
}
const DEFAULT_PREPARATION_TIMEOUT_MS = 15_000

interface VisionImageClientOptions {
  documentImpl?: Document | null
  createImageBitmapImpl?: ((
    image: ImageBitmapSource,
    options?: ImageBitmapOptions,
  ) => Promise<ImageBitmap>) | null
  createObjectUrl?: ((object: Blob | MediaSource) => string) | null
  revokeObjectUrl?: ((url: string) => void) | null
  timeoutMs?: number
}

class ImagePreparationTimeoutError extends Error {
  constructor() {
    super('Image preparation timed out before any photo was sent. Try a smaller photo or export it as JPEG.')
    this.name = 'ImagePreparationTimeoutError'
  }
}

function beforeDeadline<T>(
  promise: Promise<T>,
  deadline: number,
  lateResult?: (value: T) => void,
): Promise<T> {
  const remaining = deadline - Date.now()
  if (remaining <= 0) {
    void promise.then((value) => {
      try { lateResult?.(value) } catch { /* Late native resource cleanup is best effort. */ }
    }, () => undefined)
    return Promise.reject(new ImagePreparationTimeoutError())
  }
  return new Promise((resolve, reject) => {
    let settled = false
    const timer = globalThis.setTimeout(() => {
      settled = true
      reject(new ImagePreparationTimeoutError())
    }, remaining)
    promise.then(
      (value) => {
        if (settled) {
          try { lateResult?.(value) } catch { /* Late native resource cleanup is best effort. */ }
          return
        }
        settled = true
        globalThis.clearTimeout(timer)
        resolve(value)
      },
      (reason) => {
        if (settled) return
        settled = true
        globalThis.clearTimeout(timer)
        reject(reason)
      },
    )
  })
}

async function fileToDataUrl(file: Blob): Promise<string> {
  if (typeof FileReader !== 'undefined') {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result))
      reader.onerror = () => reject(reader.error)
      reader.readAsDataURL(file)
    })
  }
  if (typeof globalThis.btoa !== 'function') {
    throw new Error('The selected image could not be encoded in this browser.')
  }
  const bytes = new Uint8Array(await file.arrayBuffer())
  const encodedChunks: string[] = []
  const chunkBytes = 3 * 8_192
  for (let offset = 0; offset < bytes.byteLength; offset += chunkBytes) {
    let binary = ''
    for (const byte of bytes.subarray(offset, offset + chunkBytes)) {
      binary += String.fromCharCode(byte)
    }
    encodedChunks.push(globalThis.btoa(binary))
  }
  return `data:${file.type};base64,${encodedChunks.join('')}`
}

function normalizedDeclaredImageType(file: File): string | null {
  const declared = file.type.trim().toLowerCase().split(';', 1)[0]
  if (declared === 'image/jpg' || declared === 'image/pjpeg') return 'image/jpeg'
  return SUPPORTED_IMAGE_TYPES.has(declared) ? declared : null
}

function fileExtension(file: File): string {
  const match = /\.([^.]+)$/.exec(file.name.trim().toLowerCase())
  return match?.[1] ?? ''
}

function heifDecodeError(): string {
  return 'This browser could not decode this HEIC/HEIF photo. Choose it again from Photos, or export/share it as JPEG or PNG and retry.'
}

export function imageFileError(file?: File): string | null {
  if (!file) return 'Choose an image to continue.'
  if (file.size <= 0) return 'The selected image is empty.'
  if (file.size > MAX_IMAGE_SOURCE_BYTES) return 'Source photos must be 24 MB or smaller.'
  const declared = file.type.trim().toLowerCase().split(';', 1)[0]
  const extension = fileExtension(file)
  if (
    declared
    && declared !== 'application/octet-stream'
    && !normalizedDeclaredImageType(file)
    && !HEIF_IMAGE_TYPES.has(declared)
  ) return 'Use a PNG, JPEG, WebP, HEIC, or HEIF still image.'
  const knownStandard = Boolean(normalizedDeclaredImageType(file) || IMAGE_EXTENSION_TYPES[extension])
  if (knownStandard && file.size > MAX_IMAGE_UPLOAD_BYTES) {
    return 'PNG, JPEG, and WebP images must be 8 MB or smaller.'
  }
  return null
}

function isoBrand(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(...bytes.slice(offset, offset + 4)).toLowerCase()
}

function inspectHeifFileType(bytes: Uint8Array): string {
  if (bytes.length < 16) throw new Error('The selected HEIC/HEIF file type box is truncated.')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const shortSize = view.getUint32(0)
  let headerBytes = 8
  let boxBytes = shortSize
  if (shortSize === 1) {
    if (bytes.length < 24 || typeof view.getBigUint64 !== 'function') {
      throw new Error('The selected HEIC/HEIF file type box is malformed.')
    }
    const largeSize = view.getBigUint64(8)
    if (largeSize > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error('The selected HEIC/HEIF file type box is too large.')
    }
    headerBytes = 16
    boxBytes = Number(largeSize)
  } else if (shortSize === 0) {
    throw new Error('The selected HEIC/HEIF file type box has an unbounded size.')
  }
  if (
    !Number.isSafeInteger(boxBytes)
    || boxBytes < headerBytes + 8
    || boxBytes > 4_096
    || boxBytes > bytes.length
    || (boxBytes - headerBytes - 8) % 4 !== 0
  ) throw new Error('The selected HEIC/HEIF file type box is malformed.')
  if ((boxBytes - headerBytes - 8) / 4 > 128) {
    throw new Error('The selected HEIC/HEIF file declares too many compatibility brands.')
  }
  const brands = new Set([isoBrand(bytes, headerBytes)])
  for (let offset = headerBytes + 8; offset < boxBytes; offset += 4) {
    brands.add(isoBrand(bytes, offset))
  }
  const hevcStill = [...HEVC_STILL_BRANDS].some((brand) => brands.has(brand))
  const heifStill = [...HEIF_STILL_BRANDS].some((brand) => brands.has(brand))
  const sequence = [...HEIF_SEQUENCE_BRANDS].some((brand) => brands.has(brand))
  const avif = [...AVIF_BRANDS].some((brand) => brands.has(brand))
  if (Number(hevcStill) + Number(sequence) + Number(avif) > 1) {
    throw new Error('The selected ISO media file has conflicting still-image brands.')
  }
  if (avif) throw new Error('AVIF input is not supported; choose HEIC, HEIF, JPEG, PNG, or WebP.')
  if (sequence) {
    throw new Error('HEIC/HEIF image sequences are not supported; export one still image and retry.')
  }
  if (!heifStill) throw new Error('The selected ISO media file is not a supported HEIC/HEIF still image.')
  return hevcStill ? 'image/heic' : 'image/heif'
}

export function sniffImageMime(bytes: Uint8Array): string | null {
  if (
    bytes.length >= 8
    && bytes[0] === 0x89
    && bytes[1] === 0x50
    && bytes[2] === 0x4e
    && bytes[3] === 0x47
    && bytes[4] === 0x0d
    && bytes[5] === 0x0a
    && bytes[6] === 0x1a
    && bytes[7] === 0x0a
  ) return 'image/png'
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'image/jpeg'
  }
  if (
    bytes.length >= 12
    && String.fromCharCode(...bytes.slice(0, 4)) === 'RIFF'
    && String.fromCharCode(...bytes.slice(8, 12)) === 'WEBP'
  ) return 'image/webp'
  if (bytes.length >= 12 && String.fromCharCode(...bytes.slice(4, 8)) === 'ftyp') {
    return inspectHeifFileType(bytes)
  }
  return null
}

export function validatedImageMime(file: File, signature: Uint8Array): string {
  const initialError = imageFileError(file)
  if (initialError) throw new Error(initialError)
  const detected = sniffImageMime(signature)
  if (!detected) {
    throw new Error('The selected file is not a recognizable PNG, JPEG, WebP, HEIC, or HEIF still image.')
  }
  const declared = normalizedDeclaredImageType(file)
  if (declared && declared !== detected) {
    throw new Error('The selected image bytes do not match its declared file type.')
  }
  const extension = fileExtension(file)
  const declaredType = file.type.trim().toLowerCase().split(';', 1)[0]
  if (HEIF_IMAGE_TYPES.has(declaredType) && !HEIF_IMAGE_TYPES.has(detected)) {
    throw new Error('The selected image bytes do not match its declared HEIC/HEIF file type.')
  }
  if (HEIF_EXTENSIONS.has(extension) && !HEIF_IMAGE_TYPES.has(detected)) {
    throw new Error('The selected image bytes do not match its HEIC/HEIF filename extension.')
  }
  const extensionType = IMAGE_EXTENSION_TYPES[extension]
  if (!declared && extensionType && extensionType !== detected) {
    throw new Error('The selected image bytes do not match its filename extension.')
  }
  return detected
}

function canvasBlob(canvas: HTMLCanvasElement, quality: number, deadline: number): Promise<Blob> {
  const encoded = new Promise<Blob>((resolve, reject) => {
    try {
      canvas.toBlob((blob) => {
        if (!blob) reject(new Error('The browser could not canonicalize this photo.'))
        else resolve(blob)
      }, 'image/jpeg', quality)
    } catch (reason) {
      reject(reason)
    }
  })
  return beforeDeadline(encoded, deadline)
}

async function decodeHeifSource(source: Blob, options: Required<Pick<
  VisionImageClientOptions,
  'documentImpl' | 'createImageBitmapImpl' | 'createObjectUrl' | 'revokeObjectUrl'
>>, deadline: number): Promise<{
  drawable: CanvasImageSource
  width: number
  height: number
  release: () => void
}> {
  const closeBitmap = (bitmap: ImageBitmap) => {
    try { bitmap.close() } catch { /* Native decoder cleanup is best effort. */ }
  }
  if (typeof options.createImageBitmapImpl === 'function') {
    try {
      let bitmap: ImageBitmap
      try {
        bitmap = await beforeDeadline(
          options.createImageBitmapImpl(source, { imageOrientation: 'from-image' }),
          deadline,
          closeBitmap,
        )
      } catch (reason) {
        if (reason instanceof ImagePreparationTimeoutError) throw reason
        bitmap = await beforeDeadline(
          options.createImageBitmapImpl(source),
          deadline,
          closeBitmap,
        )
      }
      return {
        drawable: bitmap,
        width: bitmap.width,
        height: bitmap.height,
        release: () => closeBitmap(bitmap),
      }
    } catch (reason) {
      if (reason instanceof ImagePreparationTimeoutError) throw reason
      // Safari can decode HEIC through an HTMLImageElement even when createImageBitmap cannot.
    }
  }
  if (
    !options.documentImpl
    || typeof options.documentImpl.createElement !== 'function'
    || typeof options.createObjectUrl !== 'function'
    || typeof options.revokeObjectUrl !== 'function'
  ) throw new Error(heifDecodeError())
  const createObjectUrl = options.createObjectUrl
  const revokeObjectUrl = options.revokeObjectUrl
  let objectUrl: string | undefined
  let image: HTMLImageElement | undefined
  try {
    image = options.documentImpl.createElement('img')
    objectUrl = createObjectUrl(source)
    if (!objectUrl) throw new Error(heifDecodeError())
    image.decoding = 'async'
    image.src = objectUrl
    if (typeof image.decode !== 'function') throw new Error(heifDecodeError())
    await beforeDeadline(image.decode(), deadline)
    if (!image.naturalWidth || !image.naturalHeight) throw new Error(heifDecodeError())
    return {
      drawable: image,
      width: image.naturalWidth,
      height: image.naturalHeight,
      release: () => {
        image?.removeAttribute('src')
        if (objectUrl) {
          try { revokeObjectUrl(objectUrl) } catch { /* Native decoder cleanup is best effort. */ }
        }
      },
    }
  } catch (reason) {
    image?.removeAttribute('src')
    if (objectUrl) {
      try { revokeObjectUrl(objectUrl) } catch { /* Native decoder cleanup is best effort. */ }
    }
    if (reason instanceof ImagePreparationTimeoutError) throw reason
    throw new Error(heifDecodeError())
  }
}

function imageBlob(bytes: Uint8Array, type: string): Blob {
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  return new Blob([copy.buffer], { type })
}

async function canonicalizeHeif(
  source: Uint8Array,
  sourceType: string,
  options: Required<Pick<
    VisionImageClientOptions,
    'documentImpl' | 'createImageBitmapImpl' | 'createObjectUrl' | 'revokeObjectUrl'
  >>,
  deadline: number,
): Promise<Blob> {
  const decoded = await decodeHeifSource(imageBlob(source, sourceType), options, deadline)
  try {
    if (
      !Number.isSafeInteger(decoded.width)
      || !Number.isSafeInteger(decoded.height)
      || decoded.width < 1
      || decoded.height < 1
      || decoded.width > 8_192
      || decoded.height > 8_192
      || decoded.width * decoded.height > 50 * 1024 * 1024
    ) throw new Error('The source photo dimensions exceed the safe decode limit.')
    const scale = Math.min(
      1,
      4_096 / decoded.width,
      4_096 / decoded.height,
      Math.sqrt((16 * 1024 * 1024) / (decoded.width * decoded.height)),
    )
    let width = Math.max(1, Math.floor(decoded.width * scale))
    let height = Math.max(1, Math.floor(decoded.height * scale))
    if (!options.documentImpl) throw new Error('Browser image canonicalization is unavailable.')
    const canvas = options.documentImpl.createElement('canvas')
    for (let attempt = 0; attempt < 6; attempt += 1) {
      canvas.width = width
      canvas.height = height
      const context = canvas.getContext('2d', { alpha: false })
      if (!context) throw new Error('Browser image canonicalization is unavailable.')
      context.fillStyle = '#ffffff'
      context.fillRect(0, 0, width, height)
      context.drawImage(decoded.drawable, 0, 0, width, height)
      for (const quality of [0.9, 0.76, 0.62]) {
        const encoded = await canvasBlob(canvas, quality, deadline)
        if (encoded.type === 'image/jpeg' && encoded.size <= MAX_IMAGE_UPLOAD_BYTES) return encoded
      }
      width = Math.max(1, Math.floor(width * 0.75))
      height = Math.max(1, Math.floor(height * 0.75))
    }
    throw new Error('The canonical JPEG exceeds 8 MB after safe downscaling.')
  } finally {
    decoded.release()
  }
}

export async function fileToCanonicalImageDataUrl(
  file: File,
  clientOptions: VisionImageClientOptions = {},
): Promise<string> {
  const initialError = imageFileError(file)
  if (initialError) throw new Error(initialError)
  const timeoutMs = clientOptions.timeoutMs ?? DEFAULT_PREPARATION_TIMEOUT_MS
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60_000) {
    throw new Error('Image preparation timing limits are invalid.')
  }
  const deadline = Date.now() + timeoutMs
  const documentImpl = clientOptions.documentImpl === undefined
    ? (typeof document === 'undefined' ? null : document)
    : clientOptions.documentImpl
  const createImageBitmapImpl = clientOptions.createImageBitmapImpl === undefined
    ? globalThis.createImageBitmap
    : clientOptions.createImageBitmapImpl
  const createObjectUrl = clientOptions.createObjectUrl === undefined
    ? (typeof URL === 'undefined' ? null : URL.createObjectURL?.bind(URL) ?? null)
    : clientOptions.createObjectUrl
  const revokeObjectUrl = clientOptions.revokeObjectUrl === undefined
    ? (typeof URL === 'undefined' ? null : URL.revokeObjectURL?.bind(URL) ?? null)
    : clientOptions.revokeObjectUrl
  const options = { documentImpl, createImageBitmapImpl, createObjectUrl, revokeObjectUrl }
  const source = new Uint8Array(await beforeDeadline(file.arrayBuffer(), deadline))
  if (source.byteLength !== file.size) throw new Error('The selected image changed while it was read.')
  const mime = validatedImageMime(file, source)
  if (!HEIF_IMAGE_TYPES.has(mime) && source.byteLength > MAX_IMAGE_UPLOAD_BYTES) {
    throw new Error('PNG, JPEG, and WebP images must be 8 MB or smaller.')
  }
  const canonical = HEIF_IMAGE_TYPES.has(mime)
    ? await canonicalizeHeif(source, mime, options, deadline)
    : imageBlob(source, mime)
  if (canonical.size > MAX_IMAGE_UPLOAD_BYTES) {
    throw new Error('The canonical image must be 8 MB or smaller.')
  }
  const encoded = await beforeDeadline(fileToDataUrl(canonical), deadline)
  const comma = encoded.indexOf(',')
  if (comma < 0) throw new Error('The selected image could not be encoded safely.')
  return `data:${canonical.type};base64,${encoded.slice(comma + 1)}`
}

function decodedDataUrlBytes(value: string): number {
  const comma = value.indexOf(',')
  const encodedLength = comma >= 0 ? value.length - comma - 1 : 0
  const padding = value.endsWith('==') ? 2 : value.endsWith('=') ? 1 : 0
  return Math.max(0, Math.floor(encodedLength * 3 / 4) - padding)
}

export function imageSelectionError(existingImages: string[], files: File[]): string | null {
  if (!files.length) return 'Choose one or more images to continue.'
  if (existingImages.length + files.length > MAX_IMAGE_UPLOADS) {
    return `Attach at most ${MAX_IMAGE_UPLOADS} images to one message.`
  }
  for (const file of files) {
    const error = imageFileError(file)
    if (error) return error
  }
  return null
}

export function canonicalImageSelectionError(images: string[]): string | null {
  if (images.length > MAX_IMAGE_UPLOADS) {
    return `Attach at most ${MAX_IMAGE_UPLOADS} images to one message.`
  }
  if (images.some((image) => decodedDataUrlBytes(image) > MAX_IMAGE_UPLOAD_BYTES)) {
    return 'Each canonical image must be 8 MB or smaller.'
  }
  const totalBytes = images.reduce((total, image) => total + decodedDataUrlBytes(image), 0)
  if (totalBytes > MAX_TOTAL_IMAGE_UPLOAD_BYTES) {
    return 'Attached images must total 16 MB or smaller.'
  }
  return null
}

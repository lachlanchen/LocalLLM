import { describe, expect, it, vi } from 'vitest'
import {
  canonicalImageSelectionError,
  fileToCanonicalImageDataUrl,
  imageFileError,
  imageSelectionError,
  sniffImageMime,
  validatedImageMime,
} from './visionImageClient'

function heicBytes(): Uint8Array {
  return Uint8Array.of(
    0, 0, 0, 20,
    0x66, 0x74, 0x79, 0x70,
    0x68, 0x65, 0x69, 0x63,
    0, 0, 0, 0,
    0x6d, 0x69, 0x66, 0x31,
  )
}

function imageFile(bytes: Uint8Array, name: string, type = ''): File {
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  return new File([copy.buffer], name, { type })
}

describe('mobile vision image preparation', () => {
  it('rejects unsafe or oversized standard image attachments before reading them', () => {
    expect(imageFileError(new File(['x'], 'payload.svg', { type: 'image/svg+xml' }))).toContain('PNG')
    expect(imageFileError(new File([], 'empty.png', { type: 'image/png' }))).toContain('empty')
    const oversized = new File([new Uint8Array(8 * 1024 * 1024 + 1)], 'large.png', { type: 'image/png' })
    expect(imageFileError(oversized)).toContain('8 MB')
    expect(imageFileError(new File(['safe'], 'photo.webp', { type: 'image/webp' }))).toBeNull()
    expect(imageFileError(new File(['animated'], 'animation.gif', { type: 'image/gif' }))).toContain('PNG')
  })

  it('infers empty mobile MIME values from bytes and checks filename agreement', () => {
    const png = Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)
    const jpeg = Uint8Array.of(0xff, 0xd8, 0xff, 0xe0)
    const webp = Uint8Array.of(
      0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50,
    )

    expect(validatedImageMime(new File([png], 'iphone-export.png'), png)).toBe('image/png')
    expect(validatedImageMime(new File([jpeg], 'IMG_1234.JPG'), jpeg)).toBe('image/jpeg')
    expect(validatedImageMime(new File([webp], 'share-target'), webp)).toBe('image/webp')
    expect(() => validatedImageMime(new File([jpeg], 'wrong.png'), jpeg)).toThrow('filename')
  })

  it('recognizes HEIC still-image brands and rejects image sequences', () => {
    const heic = heicBytes()
    const sequence = Uint8Array.of(
      0, 0, 0, 16,
      0x66, 0x74, 0x79, 0x70,
      0x68, 0x65, 0x76, 0x63,
      0, 0, 0, 0,
    )

    expect(sniffImageMime(heic)).toBe('image/heic')
    expect(validatedImageMime(imageFile(heic, 'IMG_0001.HEIC'), heic)).toBe('image/heic')
    expect(() => sniffImageMime(sequence)).toThrow('sequences are not supported')
  })

  it('returns a precise fallback when no native HEIC decoder or DOM is available', async () => {
    const heic = heicBytes()
    expect(typeof document).toBe('undefined')

    await expect(fileToCanonicalImageDataUrl(
      imageFile(heic, 'IMG_0001.HEIC', 'image/heic'),
      {
        createImageBitmapImpl: null,
        createObjectUrl: null,
        revokeObjectUrl: null,
      },
    )).rejects.toThrow('could not decode this HEIC/HEIF photo')
  })

  it('bounds native HEIC decode time and closes a bitmap that resolves late', async () => {
    const heic = heicBytes()
    const close = vi.fn()
    let resolveBitmap!: (bitmap: ImageBitmap) => void
    const pendingBitmap = new Promise<ImageBitmap>((resolve) => { resolveBitmap = resolve })

    const preparation = fileToCanonicalImageDataUrl(
      imageFile(heic, 'IMG_0001.HEIC', 'image/heic'),
      {
        documentImpl: null,
        createImageBitmapImpl: () => pendingBitmap,
        createObjectUrl: null,
        revokeObjectUrl: null,
        timeoutMs: 5,
      },
    )

    await expect(preparation).rejects.toThrow('timed out before any photo was sent')
    resolveBitmap({ width: 1, height: 1, close } as unknown as ImageBitmap)
    await new Promise((resolve) => globalThis.setTimeout(resolve, 0))
    expect(close).toHaveBeenCalledOnce()
  })

  it.each([
    ['image/png', 'mobile-export.png', Uint8Array.of(
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    )],
    ['image/jpeg', 'IMG_1234.JPG', Uint8Array.of(0xff, 0xd8, 0xff, 0xe0)],
  ])('returns a canonical %s data URL for an empty mobile MIME value', async (
    expectedMime,
    filename,
    bytes,
  ) => {
    const result = await fileToCanonicalImageDataUrl(imageFile(bytes, filename))
    expect(result).toBe(`data:${expectedMime};base64,${globalThis.btoa(
      String.fromCharCode(...bytes),
    )}`)
  })

  it('canonicalizes a natively decoded HEIC photo and releases its object URL', async () => {
    const heic = heicBytes()
    const removeAttribute = vi.fn()
    const revokeObjectUrl = vi.fn()
    const decode = vi.fn(async () => undefined)
    const image = {
      decoding: '',
      src: '',
      decode,
      removeAttribute,
      naturalWidth: 2,
      naturalHeight: 1,
    }
    const drawImage = vi.fn()
    const context = {
      fillStyle: '',
      fillRect: vi.fn(),
      drawImage,
    }
    const canonicalBytes = Uint8Array.of(0xff, 0xd8, 0xff, 0xe0, 0xff, 0xd9)
    const canvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => context),
      toBlob: vi.fn((callback: BlobCallback) => {
        callback(new Blob([canonicalBytes], { type: 'image/jpeg' }))
      }),
    }
    const documentImpl = {
      createElement: vi.fn((tagName: string) => tagName === 'img' ? image : canvas),
    } as unknown as Document

    const result = await fileToCanonicalImageDataUrl(
      imageFile(heic, 'IMG_0001.HEIC', 'image/heic'),
      {
        documentImpl,
        createImageBitmapImpl: null,
        createObjectUrl: () => 'blob:test-photo',
        revokeObjectUrl,
      },
    )

    expect(result).toBe(`data:image/jpeg;base64,${globalThis.btoa(
      String.fromCharCode(...canonicalBytes),
    )}`)
    expect(decode).toHaveBeenCalledOnce()
    expect(drawImage).toHaveBeenCalledWith(image, 0, 0, 2, 1)
    expect(removeAttribute).toHaveBeenCalledWith('src')
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:test-photo')
  })

  it('revokes an HTML image URL when the native fallback decode times out', async () => {
    const heic = heicBytes()
    const removeAttribute = vi.fn()
    const revokeObjectUrl = vi.fn()
    const image = {
      decoding: '',
      src: '',
      decode: () => new Promise<void>(() => undefined),
      removeAttribute,
      naturalWidth: 0,
      naturalHeight: 0,
    }
    const documentImpl = {
      createElement: vi.fn(() => image),
    } as unknown as Document

    await expect(fileToCanonicalImageDataUrl(
      imageFile(heic, 'IMG_0001.HEIC', 'image/heic'),
      {
        documentImpl,
        createImageBitmapImpl: null,
        createObjectUrl: () => 'blob:test-photo',
        revokeObjectUrl,
        timeoutMs: 5,
      },
    )).rejects.toThrow('timed out before any photo was sent')
    expect(removeAttribute).toHaveBeenCalledWith('src')
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:test-photo')
  })

  it('enforces four ordered attachments and the 16 MB post-canonical budget', () => {
    const files = Array.from({ length: 5 }, (_, index) => (
      new File(['x'], `${index}.jpg`, { type: 'image/jpeg' })
    ))
    expect(imageSelectionError([], files)).toContain('at most 4')

    const sevenMbImage = `data:image/jpeg;base64,${'A'.repeat(Math.ceil(7 * 1024 * 1024 * 4 / 3))}`
    expect(canonicalImageSelectionError([sevenMbImage, sevenMbImage])).toBeNull()
    expect(canonicalImageSelectionError([sevenMbImage, sevenMbImage, sevenMbImage]))
      .toContain('16 MB')
  })
})

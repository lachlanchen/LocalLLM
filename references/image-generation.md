# Optional local image generation

LocalLLM has a deliberately separate, default-off text-to-image lane built around the official `Tongyi-MAI/Z-Image-Turbo` Diffusers checkpoint. It is not part of Ollama, does not accept remote image URLs, and does not start or reserve GPU memory merely because its runtime or weights are installed.

## Why Z-Image-Turbo, not FLUX.1-schnell

Both candidates are Apache-2.0 and supported by Diffusers. The comparison below uses the official Hugging Face repository metadata at the pinned revisions, not an estimate derived from a community quantization.

| Candidate | Pinned revision | Published parameter count | Diffusers-layout weight payload | Single 24 GB GPU result |
| --- | --- | ---: | ---: | --- |
| `Tongyi-MAI/Z-Image-Turbo` | `f332072aa78be7aecdf3ee76d5c247082da564a6` | 6,154,908,736 F32 parameters | 32,832,339,790 bytes on disk; loaded as BF16 | Selected. The official card says it fits within 16 GB VRAM. |
| `black-forest-labs/FLUX.1-schnell` | `741f7c3ce8b383c54771c7003378a50191e9efe9` | 11,891,178,560 BF16 transformer parameters | 33,720,953,870 bytes for the Diffusers layout | Not selected. Its transformer alone is 23,782,494,032 bytes, before two text encoders, VAE, activations, and allocator overhead. |

The current Z-Image repository totals 32,899,667,397 bytes; LocalLLM downloads only the 32,848,305,533-byte Diffusers layout plus the small README. The model is not a 4-bit quantization: the official F32 files are converted to BF16 while loading. This preserves the upstream-supported path and still leaves the second 4090 available for an LLM.

The official Z-Image example uses `num_inference_steps=9`, which results in eight DiT forward evaluations, and `guidance_scale=0.0`. LocalLLM follows those defaults. Diffusers 0.39.0 is pinned because its published wheel contains `ZImagePipeline`, `ZImageImg2ImgPipeline`, `ZImageInpaintPipeline`, and the related Z-Image pipeline modules. This lane intentionally exposes text-to-image only.

## Install without enabling

From the repository root:

```bash
scripts/setup-image-generation.sh
scripts/download-image-generation-model.sh
scripts/verify-image-generation.sh
```

The setup creates `.local/image-generation/venv` with its own fully resolved
PyTorch/CUDA wheel set. This duplicates several gigabytes already present in
the host environment, but prevents unrelated host and user-site packages from
entering the inference process. The bootstrap interpreter must be exactly
Python 3.10.13 because that version and its `lib/python3.10` layout are part of
the API, Bubblewrap, and marker attestation; setup fails before creating a venv
for any other interpreter. Installation uses `pip --require-hashes`
against `tools/image-generation/requirements.lock.txt`. Only after exact import
and version checks does it write a bounded runtime marker containing Python
3.10.13, every resolved package version, and current SHA-256 values for both
the readable requirements file and hashed lock. API readiness recomputes and
requires that exact attestation, so a stale or drifted environment fails closed.
The worker mounts that marker read-only, hides the base Conda `site-packages`,
and rechecks all 64 installed distribution versions before importing the model.
The downloader resolves the exact model commit into
`.local/models/image-generation/z-image-turbo-f332072a`, accepts only the
Diffusers data formats, requires 100 GiB to remain free, checks the exact
seven-file weight manifest and every official LFS SHA-256, and writes a
revision/license/hash marker only after validation.

`verify-image-generation.sh` is static: it imports the pinned pipeline, checks the model layout and revision marker, and reports free memory on the selected GPU. It does not load weights or generate an image.

An operator-authorized real smoke uses the configured physical GPU, writes one fixed 512-pixel test artifact under `.local/image-generation/smoke-data`, reports measured latency/peak allocation, and always unloads the worker:

```bash
LOCALLLM_IMAGE_GENERATION_GPU=0 scripts/smoke-image-generation.sh
```

Installing and downloading do **not** enable the API. To opt in, set:

```dotenv
LOCALLLM_IMAGE_GENERATION_ENABLED=true
LOCALLLM_IMAGE_GENERATION_GPU=0
LOCALLLM_IMAGE_GENERATION_TIMEOUT_SECONDS=300
```

Then restart only the LocalLLM API through the normal user-service workflow.
The portable configuration default remains physical GPU 0. Check `ollama ps`
and `nvidia-smi` before choosing: on this workstation the measured one-card
Ollama load occupies physical GPU 1, so GPU 0 is the current image-generation
choice. The sandbox mounts only `/dev/nvidiactl`, `/dev/nvidia-uvm`, and the
configured physical card node; that card becomes logical `cuda:0` inside the
worker. The other GPU node, host runtime sockets, home directory, repository,
and host PID namespace are absent. Although the upstream card describes a
sub-16-GB model path, this complete pinned runtime measured a
21,352,528,384-byte PyTorch peak for the real 512px smoke. LocalLLM therefore
requires `nvidia-smi` to report at least 22 GiB free before a cold worker start;
status marks the lane unavailable and create fails closed below that threshold.
A resident Ollama runner, desktop renderer, or any other CUDA application counts
toward that physical-card capacity check.
A worker that already owns the model can accept its next serialized job without
paying the cold-load allocation again. This is a strong least-visibility local
boundary, not a claim that a GPU driver or kernel exploit is impossible.

## Mounted service

The main application mounts `localllm.image_generation` under `/api/images`.
Its lifespan lazily owns and shuts down the manager, and it starts no subprocess
until an enabled, authenticated job is submitted. Job creation is capped at an
8 KiB encoded request body by both the application's global limiter and the
route's bounded stream reader. The unload endpoint requires an empty body.

When the API itself runs inside the hardened LocalLLM user unit, Ubuntu blocks
a nested unprivileged Bubblewrap user namespace from that unit's mount
namespace. In that environment the API asks the same user systemd manager to
start the fixed Bubblewrap command in a uniquely named, short-lived transient
unit. The unit is runtime-capped, collected after exit, and explicitly stopped
on cancellation, timeout, unload, or application shutdown. The worker still
enters the fresh-root Bubblewrap boundary described below; this launch path does
not replace it or relax the permanent API service.

## API

The status endpoint is safe while disabled:

```bash
curl http://127.0.0.1:8008/api/images/status
```

Only status is public within the application's loopback/origin boundary.
Creating, listing, polling, reading, and deleting jobs, plus explicitly
unloading the worker, all require the local API key. Only a text prompt and
bounded scalar generation settings are accepted; extra fields such as
`image_url` are rejected.

```bash
curl -sS http://127.0.0.1:8008/api/images/jobs \
  -H 'Content-Type: application/json' \
  -H 'X-LocalLLM-Key: local-dev-key' \
  -d '{
    "prompt": "A cheerful solar-powered robot workshop, bright modern illustration",
    "width": 1024,
    "height": 1024,
    "steps": 9,
    "seed": 42,
    "output_format": "png"
  }'
```

List up to 128 current and retained records with `GET /api/images/jobs`, then
poll `GET /api/images/jobs/{id}`. A successful response supplies
`/api/images/jobs/{id}/image`. `DELETE /api/images/jobs/{id}` cancels a
queued/running job, terminates and reaps the worker when necessary, and removes
that job's image and metadata. Authenticated `POST /api/images/unload` takes an
empty body and immediately releases a warm worker without deleting completed
images; it returns HTTP 409 rather than interrupting an active job.

## Playground panel

Image Studio is mounted as a collapsed optional panel above the Playground
composer. A status/list probe runs on page mount even while it stays collapsed;
the public status half discovers a warm worker after reload so chat remains
serialized correctly, and the keyed list half restores saved results. Neither
loads weights. The panel reports whether the runtime, exact
checkpoint, operator enablement, and worker are ready; restores the bounded
saved-output list; accepts only the text-to-image settings described above; and
exposes progress, cancel, delete, download, and **Release GPU** controls. **Use
in chat** first calls the authenticated unload endpoint and verifies the worker
is no longer resident. Only then does it fetch the generated file with the API
key, validate the private blob against the chat attachment boundary, and stage
it for the next vision turn. Preview and download also use an authenticated
in-memory blob URL rather than exposing an unkeyed image URL. An authentication,
network, or unload failure leaves the image unattached. It never silently sends
a chat request.

## Safety and resource boundaries

- The main application remains fixed to `127.0.0.1:8008`; this router adds no listener.
- The feature is false by default; every job/output route and mutation requires the configured LocalLLM key, while status remains loopback-public.
- No URL, upload, image-to-image, arbitrary model ID, arbitrary model path, or arbitrary output path is accepted.
- The persistent worker receives requests over bounded stdin JSON and starts from a cleared environment with an explicit non-secret allowlist. Bubblewrap supplies a fresh root instead of bind-mounting `/`, read-only system/Python/model/script mounts, private `/tmp` and `/run`, exact writable cache/output/work mounts, and new network, PID, IPC, UTS, user-when-supported, and cgroup-when-supported namespaces. It drops all capabilities and starts a new session. Only NVIDIA control/UVM plus the selected physical GPU node are mounted; the other GPU and Docker/desktop/runtime sockets are absent. Hugging Face and Transformers offline flags provide a second independent no-download control.
- The model class is imported from pinned Diffusers; `trust_remote_code` is not used. Weights are safetensors from the pinned official revision.
- Width and height are 512–1536, multiples of 64, and capped at 1,572,864 total pixels. Scheduler steps are 4–12. Prompts are at most 2,000 characters.
- Generation concurrency is exactly one, the pending queue is four, and the whole first-load-plus-generation operation has a 60–900-second operator-configured timeout.
- GPU readiness includes a bounded physical-card free-memory probe. Cold submit and the queued-job handoff both require at least 22 GiB free, preventing a retained Ollama runner on the configured card from becoming a predictable CUDA OOM collision. The public status response exposes current free bytes and the threshold so the UI can explain the block.
- Output is RGB PNG or JPEG only, written without caller metadata to a mode-0700 fixed directory. Files are mode 0600, signature/dimensions are verified before publication, each file is capped at 32 MiB, and the directory is capped at 128 images/1 GiB. A prompt-free, bounded mode-0600 JSON sidecar is atomically published for each successful image. Valid retained images are never evicted merely for quota pressure; new work is rejected until the user deletes an output.
- During application lifespan startup—and before authenticated create/delete if startup cleanup could not complete—bounded reconciliation removes stale app-owned `.part` files and unusable app-owned image artifacts, discards metadata whose image is gone, reloads valid sidecar/image pairs, and reconstructs a bounded record for valid pre-sidecar images. `GET` status/list/job/image operations are read-only. Reconstructed legacy records expose `settings_known: false` because their original seed and scheduler settings cannot be proven. Symlinks, hard links, and unknown directory entries fail closed instead of being deleted.
- Cancellation or timeout terminates the worker process group, waits, escalates to `SIGKILL` after five seconds, reaps it, and removes partial output. A later request starts a clean worker.
- A successfully warmed worker unloads automatically after 120 idle seconds. A new queued job cancels that timer. The UI also exposes **Release GPU**, and reports itself busy to its parent while the worker remains warm, so a dual-GPU LLM request need not collide with resident image weights.

## Operational limitations

The first job after API start, cancellation, explicit release, or idle unload
must load about 32.8 GB of on-disk weights and is slower than a request made
while the worker is warm. Active/failed control records are process-local;
successful image records and their non-prompt generation settings are restored
from `data/image-generation` after a restart. Older valid images created before
sidecars existed are recovered with their dimensions and format, marked as
having unknown original settings, and remain available through the normal
list/read/delete API. Direct filesystem maintenance should be done only while
the API is stopped.

This is a local text-to-image service, not a content-safety classifier, image editor, or hosted-provider parity claim. Output quality and latency depend on prompt, resolution, driver, PyTorch, PCIe topology, and current GPU load. The official “sub-second” statement applies to an H800 and must not be projected onto a 4090 without measurement.

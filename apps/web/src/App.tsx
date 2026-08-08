import {
  Activity,
  Aperture,
  ArrowRight,
  Binary,
  BookOpen,
  Boxes,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleStop,
  Clipboard,
  Cloud,
  Code2,
  Cpu,
  Download,
  ExternalLink,
  FileCode2,
  FlaskConical,
  Gauge,
  GitBranch,
  Globe2,
  HardDrive,
  ImagePlus,
  KeyRound,
  Layers3,
  LoaderCircle,
  Menu,
  MessageCircleMore,
  MonitorCog,
  Paperclip,
  Play,
  RefreshCw,
  Search,
  Send,
  Server,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  Trash2,
  UploadCloud,
  WandSparkles,
  X,
  type LucideIcon,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, fileToDataUrl, formatBytes, pullModel, streamChat } from './api'
import {
  chooseAvailableAlias,
  isAliasInstalled,
  modelChoiceStatuses,
  type ModelKind,
} from './modelAvailability'
import type {
  BinaryMetadata,
  CatalogResponse,
  ChatMessage,
  McpInvestigationResult,
  McpStatus,
  ModelInfo,
  ResearchTask,
  SystemStatus,
  ViewId,
} from './types'

const NAV_ITEMS: Array<{ id: ViewId; label: string; hint: string; icon: LucideIcon }> = [
  { id: 'chat', label: 'Playground', hint: 'Text & images', icon: MessageCircleMore },
  { id: 'vision', label: 'Vision lab', hint: 'Images & OCR', icon: Aperture },
  { id: 'research', label: 'Deep research', hint: 'Search & cite', icon: Globe2 },
  { id: 'models', label: 'Model shelf', hint: 'Pull & compare', icon: Boxes },
  { id: 'reverse', label: 'Binary studio', hint: 'Ghidra & MCP', icon: Binary },
  { id: 'api', label: 'API desk', hint: 'OpenAI format', icon: Code2 },
]

const PROMPTS = [
  { icon: FlaskConical, title: 'Explain a paper', text: 'Explain mixture-of-experts routing as if I am designing a real-time robotics system.' },
  { icon: Code2, title: 'Pair program', text: 'Design a robust Python serial protocol with retries, framing, and CRC for an ESP32.' },
  { icon: BrainCircuit, title: 'Think deeply', text: 'Compare three architectures for a private local research assistant and recommend one.' },
]

function uid() {
  return crypto.randomUUID()
}

function Markdown({ children }: { children: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
}

function Logo() {
  return (
    <div className="brand">
      <div className="brand-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <div>
        <strong>LocalLLM</strong>
        <small>STUDIO / 01</small>
      </div>
    </div>
  )
}

function StatusDot({ ok }: { ok?: boolean }) {
  return <span className={`status-dot ${ok ? 'is-ok' : 'is-off'}`} />
}

function AppHeader({ status, onRefresh }: { status: SystemStatus | null; onRefresh: () => void }) {
  const gpuLabel = status?.gpu.ok
    ? `${status.gpu.devices.length} GPU${status.gpu.devices.length === 1 ? '' : 's'} ready`
    : 'GPU needs attention'
  return (
    <header className="topbar">
      <div className="crumb">
        <span>PRIVATE COMPUTE</span>
        <ArrowRight size={13} />
        <strong>CONTROL ROOM</strong>
      </div>
      <div className="system-pills">
        <button className="system-pill" onClick={onRefresh} title="Refresh system status">
          <StatusDot ok={status?.ollama.ok} />
          <span>Ollama {status?.ollama.version ?? 'offline'}</span>
        </button>
        <div className="system-pill" title={status?.gpu.diagnosis ?? gpuLabel}>
          <StatusDot ok={status?.gpu.ok} />
          <span>{gpuLabel}</span>
        </div>
        <a className="icon-button" href="https://github.com/lachlanchen/LocalLLM" target="_blank" rel="noreferrer" aria-label="GitHub repository">
          <GitBranch size={18} />
        </a>
      </div>
    </header>
  )
}

function Sidebar({ view, setView, open, setOpen }: { view: ViewId; setView: (view: ViewId) => void; open: boolean; setOpen: (value: boolean) => void }) {
  return (
    <>
      <button className="mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle navigation"><Menu size={20} /></button>
      <aside className={`sidebar ${open ? 'is-open' : ''}`}>
        <Logo />
        <nav>
          <span className="nav-eyebrow">WORKSPACES</span>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            return (
              <button
                key={item.id}
                data-testid={`nav-${item.id}`}
                className={`nav-item ${view === item.id ? 'is-active' : ''}`}
                onClick={() => { setView(item.id); setOpen(false) }}
              >
                <Icon size={18} strokeWidth={1.8} />
                <span><strong>{item.label}</strong><small>{item.hint}</small></span>
                {view === item.id && <span className="nav-tick" />}
              </button>
            )
          })}
        </nav>
        <div className="privacy-card">
          <div className="privacy-icon"><ShieldCheck size={18} /></div>
          <div><strong>Private by design</strong><p>Prompts stay on this machine. Web access only runs inside Research.</p></div>
        </div>
        <div className="sidebar-footer"><span>LOCAL FIRST</span><span className="footer-line" /><span>v0.1</span></div>
      </aside>
      {open && <button className="sidebar-scrim" onClick={() => setOpen(false)} aria-label="Close navigation" />}
    </>
  )
}

function ModelSelect({
  model,
  setModel,
  catalog,
  kind = 'text',
  testId,
}: {
  model: string
  setModel: (model: string) => void
  catalog: CatalogResponse | null
  kind?: ModelKind
  testId?: string
}) {
  const choices = modelChoiceStatuses(catalog, kind)
  const hasInstalledChoice = choices.some((choice) => choice.installed)
  const selectedIsInstalled = isAliasInstalled(catalog, model)
  const statusLabel = !catalog
    ? 'Checking installed models'
    : selectedIsInstalled
      ? 'Installed model ready'
      : `No ${kind} model installed`
  return (
    <label className={`model-select ${selectedIsInstalled ? 'is-ready' : 'is-unavailable'}`} title={statusLabel}>
      <Sparkles size={15} />
      <select
        data-testid={testId ?? `${kind}-model-select`}
        aria-label={`${kind === 'vision' ? 'Vision' : 'Text'} model`}
        value={model}
        onChange={(event) => setModel(event.target.value)}
        disabled={!catalog || !hasInstalledChoice}
      >
        {choices.map((choice) => (
          <option key={choice.alias} value={choice.alias} disabled={choice.installed === false}>
            {choice.label}{choice.installed === false ? ' · download required' : ''}
          </option>
        ))}
      </select>
      <ChevronDown size={14} />
      <span className="model-ready-dot" aria-label={statusLabel} />
    </label>
  )
}

function ModelGateNote({ catalog, kind }: { catalog: CatalogResponse | null; kind: ModelKind }) {
  if (catalog && chooseAvailableAlias(catalog, '', kind)) return null
  return (
    <p className="model-gate-note" role="status">
      {catalog ? <Download size={13} /> : <LoaderCircle className="spin" size={13} />}
      {catalog
        ? `Install a ${kind === 'vision' ? 'vision' : 'text'} model from Model shelf to continue.`
        : 'Checking the local model catalog…'}
    </p>
  )
}

function HeroTitle({ eyebrow, title, accent, copy }: { eyebrow: string; title: string; accent?: string; copy: string }) {
  return (
    <div className="view-heading">
      <span className="eyebrow"><i />{eyebrow}</span>
      <h1>{title} {accent && <em>{accent}</em>}</h1>
      <p>{copy}</p>
    </div>
  )
}

function ChatView({ catalog, initialPrompt }: { catalog: CatalogResponse | null; initialPrompt?: string }) {
  const [model, setModel] = useState('localllm-fast')
  const [input, setInput] = useState(initialPrompt ?? '')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [image, setImage] = useState<string | undefined>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const abortRef = useRef<AbortController | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const textModel = chooseAvailableAlias(catalog, model, 'text')
  const visionModel = chooseAvailableAlias(catalog, 'localllm-vision', 'vision')
  const activeModel = image ? visionModel : textModel

  useEffect(() => endRef.current?.scrollIntoView({ behavior: 'smooth' }), [messages])
  useEffect(() => {
    if (textModel && textModel !== model) setModel(textModel)
  }, [model, textModel])

  const send = useCallback(async () => {
    const text = input.trim()
    if ((!text && !image) || busy || !activeModel) return
    const user: ChatMessage = { id: uid(), role: 'user', content: text || 'Describe this image.', image }
    const assistantId = uid()
    const next = [...messages, user]
    setMessages([...next, { id: assistantId, role: 'assistant', content: '', pending: true, model: activeModel }])
    setInput('')
    setImage(undefined)
    setBusy(true)
    setError('')
    const controller = new AbortController()
    abortRef.current = controller
    try {
      await streamChat(next, activeModel, (token) => {
        setMessages((current) => current.map((message) =>
          message.id === assistantId ? { ...message, content: message.content + token, pending: false } : message,
        ))
      }, controller.signal)
    } catch (reason) {
      if ((reason as Error).name !== 'AbortError') {
        setError(reason instanceof Error ? reason.message : 'The local model could not respond.')
      }
    } finally {
      setBusy(false)
      abortRef.current = null
      setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, pending: false } : message))
    }
  }, [activeModel, busy, image, input, messages])

  const attach = async (file?: File) => {
    if (!file || !file.type.startsWith('image/')) return
    setImage(await fileToDataUrl(file))
  }

  return (
    <section className="chat-view">
      {messages.length === 0 ? (
        <div className="chat-empty">
          <div className="hero-orbit"><div className="orbit-core"><BrainCircuit size={31} /></div><span /><span /><span /></div>
          <HeroTitle eyebrow="LOCAL INTELLIGENCE, YOUR RULES" title="Think locally." accent="Build freely." copy="One private workspace for language, code, images, research, and the strange little experiments you have been meaning to try." />
          <div className="prompt-grid">
            {PROMPTS.map((prompt) => {
              const Icon = prompt.icon
              return <button key={prompt.title} onClick={() => setInput(prompt.text)}><Icon size={19} /><strong>{prompt.title}</strong><span>{prompt.text}</span><ArrowRight size={16} /></button>
            })}
          </div>
        </div>
      ) : (
        <div className="conversation">
          <div className="conversation-title"><span className="eyebrow"><i />PRIVATE THREAD</span><h2>Untitled experiment</h2></div>
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="avatar">{message.role === 'user' ? 'YOU' : <Sparkles size={16} />}</div>
              <div className="message-body">
                <span className="message-author">{message.role === 'user' ? 'You' : 'LocalLLM'}<small>{message.role === 'assistant' ? message.model ?? model : 'just now'}</small></span>
                {message.image && <img src={message.image} alt="User attachment" />}
                {message.pending && !message.content ? <div className="typing"><i /><i /><i /></div> : <Markdown>{message.content}</Markdown>}
              </div>
            </article>
          ))}
          <div ref={endRef} />
        </div>
      )}
      <div className="composer-wrap">
        {error && <div className="inline-error"><Activity size={15} />{error}</div>}
        {image && <div className="attachment-preview"><img src={image} alt="Ready to send" /><span>Image ready</span><button onClick={() => setImage(undefined)}><X size={15} /></button></div>}
        <div className="composer">
          <textarea data-testid="chat-input" value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void send() }
          }} placeholder="Ask, create, compare, or explore…" rows={1} />
          <div className="composer-actions">
            <div className="composer-left">
              <ModelSelect model={model} setModel={setModel} catalog={catalog} />
              <input ref={fileRef} type="file" accept="image/*" hidden onChange={(event) => void attach(event.target.files?.[0])} />
              <button className="tool-button" onClick={() => fileRef.current?.click()}><Paperclip size={16} /><span>Image</span></button>
            </div>
            {busy ? <button data-testid="chat-send" data-status="running" className="send-button stop" onClick={() => abortRef.current?.abort()}><CircleStop size={18} /></button> : <button data-testid="chat-send" data-status="ready" className="send-button" onClick={() => void send()} disabled={(!input.trim() && !image) || !activeModel}><Send size={18} /></button>}
          </div>
        </div>
        {!activeModel
          ? <ModelGateNote catalog={catalog} kind={image ? 'vision' : 'text'} />
          : <p className="composer-note"><ShieldCheck size={13} /> Generated locally. Verify important outputs.</p>}
      </div>
    </section>
  )
}

function VisionView({ catalog }: { catalog: CatalogResponse | null }) {
  const [model, setModel] = useState('localllm-vision')
  const [image, setImage] = useState<string>()
  const [prompt, setPrompt] = useState('Describe this image precisely. Read all visible text and call out anything uncertain.')
  const [answer, setAnswer] = useState('')
  const [busy, setBusy] = useState(false)
  const [drag, setDrag] = useState(false)
  const availableModel = chooseAvailableAlias(catalog, model, 'vision')

  useEffect(() => {
    if (availableModel && availableModel !== model) setModel(availableModel)
  }, [availableModel, model])

  const chooseFile = async (file?: File) => {
    if (file?.type.startsWith('image/')) setImage(await fileToDataUrl(file))
  }
  const analyze = async () => {
    if (!image || busy || !availableModel) return
    setAnswer('')
    setBusy(true)
    const message: ChatMessage = { id: uid(), role: 'user', content: prompt, image }
    try { await streamChat([message], availableModel, (token) => setAnswer((current) => current + token)) }
    catch (error) { setAnswer(`Unable to analyze locally: ${error instanceof Error ? error.message : String(error)}`) }
    finally { setBusy(false) }
  }
  return (
    <section className="padded-view">
      <HeroTitle eyebrow="MULTIMODAL WORKBENCH" title="See more." accent="Send nothing." copy="Inspect screenshots, diagrams, documents, hardware photos, and visual bugs with a vision model that stays on your GPUs." />
      <div className="vision-layout">
        <label className={`dropzone ${image ? 'has-image' : ''} ${drag ? 'is-dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={(event) => { event.preventDefault(); setDrag(false); void chooseFile(event.dataTransfer.files[0]) }}>
          <input type="file" accept="image/*" hidden onChange={(event) => void chooseFile(event.target.files?.[0])} />
          {image ? <><img src={image} alt="Vision input" /><span className="replace-image"><RefreshCw size={15} /> Replace image</span></> : <div className="dropzone-empty"><div><ImagePlus size={28} /></div><strong>Drop an image into the lab</strong><p>PNG, JPEG, WEBP · screenshots welcome</p><span>CHOOSE IMAGE</span></div>}
        </label>
        <div className="vision-panel">
          <div className="panel-kicker"><Aperture size={17} /><span>INSPECTION BRIEF</span></div>
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={5} />
          <div className="vision-controls"><ModelSelect model={model} setModel={setModel} catalog={catalog} kind="vision" /><button className="primary-button" disabled={!image || busy || !availableModel} onClick={() => void analyze()}>{busy ? <LoaderCircle className="spin" size={17} /> : <WandSparkles size={17} />} Analyze</button></div>
          <ModelGateNote catalog={catalog} kind="vision" />
          <div className={`vision-result ${answer ? 'has-answer' : ''}`}>
            {answer ? <Markdown>{answer}</Markdown> : <div><Sparkles size={21} /><strong>Your analysis will appear here</strong><p>Try OCR, UI critique, circuit inspection, chart reading, or visual question answering.</p></div>}
          </div>
        </div>
      </div>
    </section>
  )
}

function ResearchView({ catalog }: { catalog: CatalogResponse | null }) {
  const [question, setQuestion] = useState('')
  const [model, setModel] = useState('localllm-deep')
  const [task, setTask] = useState<ResearchTask | null>(null)
  const [error, setError] = useState('')
  const [cancelling, setCancelling] = useState(false)
  const availableModel = chooseAvailableAlias(catalog, model, 'text')

  useEffect(() => {
    if (availableModel && availableModel !== model) setModel(availableModel)
  }, [availableModel, model])

  useEffect(() => {
    if (!task || !['queued', 'running'].includes(task.status)) return
    const timer = window.setInterval(() => {
      void api.research(task.id).then(setTask).catch((reason) => setError(String(reason)))
    }, 1500)
    return () => window.clearInterval(timer)
  }, [task])

  const start = async () => {
    if (question.trim().length < 8 || !availableModel) return
    setError('')
    try { setTask(await api.createResearch(question.trim(), availableModel)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  const reset = async () => {
    if (!task || cancelling) return
    setError('')
    if (['queued', 'running'].includes(task.status)) {
      setCancelling(true)
      try { await api.cancelResearch(task.id) }
      catch (reason) {
        setError(reason instanceof Error ? reason.message : String(reason))
        setCancelling(false)
        return
      }
      setCancelling(false)
    }
    setTask(null)
    setQuestion('')
  }
  return (
    <section className="padded-view research-view">
      <HeroTitle eyebrow="AGENTIC WEB RESEARCH" title="Search widely." accent="Cite carefully." copy="Your local model plans the investigation, searches the open web, reads sources, and produces a traceable report with uncertainty kept visible." />
      {!task ? (
        <div className="research-launch">
          <div className="research-textarea-wrap"><Search size={23} /><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What do you want to understand deeply?" rows={4} /></div>
          <div className="research-options"><ModelSelect model={model} setModel={setModel} catalog={catalog} /><div className="depth-control"><span>DEPTH</span><button className="is-active">Balanced</button><button disabled>Exhaustive</button></div><button className="primary-button large" onClick={() => void start()} disabled={question.trim().length < 8 || !availableModel}><Globe2 size={18} /> Begin research</button></div>
          <ModelGateNote catalog={catalog} kind="text" />
          <div className="research-explain"><div><span>01</span><strong>Plan</strong><p>Generate distinct search angles.</p></div><ArrowRight size={17} /><div><span>02</span><strong>Read</strong><p>Extract clean evidence from sources.</p></div><ArrowRight size={17} /><div><span>03</span><strong>Synthesize</strong><p>Write a cited, uncertainty-aware report.</p></div></div>
        </div>
      ) : (
        <div className="research-run">
          <aside className="research-progress-card">
            <span className="eyebrow"><i />LIVE RUN</span><h3>{task.question}</h3>
            <div className="progress-ring" style={{ '--progress': `${task.progress * 3.6}deg` } as React.CSSProperties}><div><strong>{task.progress}%</strong><span>{task.status}</span></div></div>
            <div className="progress-bar"><span style={{ width: `${task.progress}%` }} /></div><p>{task.stage}</p>
            {task.queries.length > 0 && <div className="query-list"><strong>SEARCH PLAN</strong>{task.queries.map((query) => <span key={query}><Search size={12} />{query}</span>)}</div>}
            <button className="ghost-button" onClick={() => void reset()} disabled={cancelling}>{cancelling ? <LoaderCircle className="spin" size={15} /> : <Trash2 size={15} />} {['queued', 'running'].includes(task.status) ? 'Cancel & start new' : 'New research'}</button>
          </aside>
          <div className="research-report">
            {task.status === 'complete' ? <><div className="report-header"><div><span>RESEARCH REPORT</span><h2>{task.question}</h2></div><button className="icon-text-button" onClick={() => navigator.clipboard.writeText(task.report)}><Clipboard size={15} /> Copy</button></div><div className="markdown-report"><Markdown>{task.report}</Markdown></div><div className="source-grid">{task.sources.map((source, index) => <a key={source.url} href={source.url} target="_blank" rel="noreferrer"><span>{String(index + 1).padStart(2, '0')}</span><div><strong>{source.title}</strong><small>{new URL(source.url).hostname}</small></div><ExternalLink size={14} /></a>)}</div></> : ['failed', 'cancelled'].includes(task.status) ? <div className="empty-report error"><Activity size={25} /><strong>{task.status === 'cancelled' ? 'Research cancelled' : 'Research stopped'}</strong><p>{task.error ?? 'This run was stopped before completion.'}</p></div> : <div className="empty-report"><div className="research-loader"><Globe2 size={26} /><i /><i /></div><strong>{task.stage}</strong><p>The local agent is working. You can leave this view and return.</p></div>}
          </div>
        </div>
      )}
      {error && <div className="inline-error"><Activity size={15} />{error}</div>}
    </section>
  )
}

interface ModelPullState { value: number; status: string; failed?: boolean }

function ModelCard({ model, progress, onPull }: { model: ModelInfo; progress?: ModelPullState; onPull: (model: string) => void }) {
  const color = model.modalities.includes('image') ? 'violet' : model.modalities.includes('embedding') ? 'teal' : model.family.includes('30B') ? 'orange' : model.family.includes('8B') ? 'teal' : 'yellow'
  return (
    <article className={`model-card ${color}`}>
      <div className="model-top"><span className="model-tier">{model.tier}</span>{model.recommended && <span className="recommended"><Sparkles size={12} /> PICK</span>}</div>
      <div className="model-icon">{model.modalities.includes('image') ? <Aperture size={25} /> : model.modalities.includes('embedding') ? <Search size={25} /> : model.family.includes('30B') ? <Layers3 size={25} /> : <BrainCircuit size={25} />}</div>
      <h3>{model.family}</h3><code>{model.quantization}</code><p>{model.role}</p>
      <div className="model-stats"><div><strong>{model.size_gb} GB</strong><span>ON DISK</span></div><div><strong>{Math.round(model.context / 1024)}K</strong><span>CONTEXT</span></div><div><strong>{model.modalities.includes('image') ? 'VISION' : model.modalities.includes('embedding') ? 'EMBED' : 'TEXT'}</strong><span>MODE</span></div></div>
      {progress?.failed ? <div className="download-progress is-failed"><span>{progress.status}</span><button className="download-button" onClick={() => onPull(model.id)}><RefreshCw size={14} /> Retry download</button></div> : progress ? <div className="download-progress"><div><span>{progress.status}</span><strong>{progress.value}%</strong></div><div className="progress-bar"><span style={{ width: `${progress.value}%` }} /></div></div> : <button className={model.installed ? 'installed-button' : 'download-button'} onClick={() => !model.installed && onPull(model.id)} disabled={model.installed}>{model.installed ? <><Check size={16} /> Installed</> : <><Download size={16} /> Pull model</>}</button>}
    </article>
  )
}

function ModelsView({ catalog, refresh }: { catalog: CatalogResponse | null; refresh: () => Promise<void> }) {
  const [pulls, setPulls] = useState<Record<string, ModelPullState>>({})
  const [filter, setFilter] = useState<'all' | 'text' | 'vision' | 'embedding'>('all')
  const pull = async (model: string) => {
    setPulls((current) => ({ ...current, [model]: { value: 0, status: 'Starting download' } }))
    try {
      await pullModel(model, (value, status) => setPulls((current) => ({ ...current, [model]: { value, status } })))
      await refresh()
      setPulls((current) => { const next = { ...current }; delete next[model]; return next })
    } catch (error) {
      setPulls((current) => ({ ...current, [model]: { value: 0, status: error instanceof Error ? error.message : 'Download failed', failed: true } }))
    }
  }
  const models = catalog?.models.filter((model) => {
    if (filter === 'all') return true
    if (filter === 'vision') return model.modalities.includes('image')
    if (filter === 'embedding') return model.modalities.includes('embedding')
    return !model.modalities.includes('image') && !model.modalities.includes('embedding')
  }) ?? []
  const installed = catalog?.models.filter((model) => model.installed).length ?? 0
  return (
    <section className="padded-view models-view">
      <div className="models-header"><HeroTitle eyebrow="CURATED FOR DUAL RTX 4090" title="A shelf with" accent="a point of view." copy="Keep Q4 for speed and Q8 for fidelity. Every preset maps to a named Ollama tag with a visible quantization—no mystery choices." /><div className="library-summary"><div><HardDrive size={19} /><span><strong>{catalog?.planned_download_gb ?? '—'} GB</strong> complete set</span></div><div><Check size={19} /><span><strong>{installed}/{catalog?.models.length ?? 0}</strong> installed</span></div></div></div>
      {catalog?.ollama?.ok === false && <div className="inline-error"><Activity size={15} />Model runtime unavailable: {catalog.ollama.error ?? 'Ollama did not answer.'}</div>}
      <div className="filter-row"><div>{(['all', 'text', 'vision', 'embedding'] as const).map((item) => <button key={item} className={filter === item ? 'is-active' : ''} onClick={() => setFilter(item)}>{item}</button>)}</div><button className="icon-text-button" onClick={() => void refresh()}><RefreshCw size={15} /> Refresh</button></div>
      <div className="model-grid">{models.map((model) => <ModelCard key={model.id} model={model} progress={pulls[model.id]} onPull={(id) => void pull(id)} />)}</div>
      <div className="quant-note"><Gauge size={24} /><div><strong>Why both quantizations?</strong><p>Q4_K_M is the daily driver: less VRAM, larger KV cache, faster startup. Q8_0 is the comparison lane when small accuracy differences matter. The 30B Q8 spans both cards; the smaller models fit on one.</p></div></div>
    </section>
  )
}

function McpInvestigator({
  catalog,
  model,
  setModel,
}: {
  catalog: CatalogResponse | null
  model: string
  setModel: (model: string) => void
}) {
  const [status, setStatus] = useState<McpStatus | null>(null)
  const [statusBusy, setStatusBusy] = useState(true)
  const [statusError, setStatusError] = useState('')
  const [binaryName, setBinaryName] = useState('')
  const [question, setQuestion] = useState('Which input paths are security-relevant, and where are their bounds validated?')
  const [result, setResult] = useState<McpInvestigationResult | null>(null)
  const [investigationError, setInvestigationError] = useState('')
  const [investigating, setInvestigating] = useState(false)
  const availableModel = chooseAvailableAlias(catalog, model, 'text')
  const binaries = status?.binaries ?? []
  const mutationCount = Array.isArray(status?.mutation_tools_blocked)
    ? status.mutation_tools_blocked.length
    : 8

  const refresh = useCallback(async () => {
    setStatusBusy(true)
    try {
      const next = await api.mcpStatus()
      setStatus(next)
      setStatusError(next.ok ? '' : next.error ?? 'The read-only MCP bridge is unavailable.')
    } catch (reason) {
      setStatus(null)
      setStatusError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setStatusBusy(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 20000)
    return () => window.clearInterval(timer)
  }, [refresh])

  useEffect(() => {
    if (!status?.ok) return
    if (binaries.some((binary) => binary.name === binaryName)) return
    setBinaryName(binaries[0]?.name ?? '')
  }, [binaries, binaryName, status?.ok])

  const investigate = async () => {
    if (!status?.ok || !binaryName || question.trim().length < 8 || !availableModel || investigating) return
    setInvestigating(true)
    setInvestigationError('')
    setResult(null)
    try {
      setResult(await api.investigateMcp(binaryName, question.trim(), availableModel))
    } catch (reason) {
      setInvestigationError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setInvestigating(false)
    }
  }

  const selectedBinary = binaries.find((binary) => binary.name === binaryName)

  return (
    <section className={`mcp-investigator ${status?.ok ? 'is-live' : ''}`} data-testid="mcp-investigator">
      <div className="mcp-investigator-header">
        <div className="mcp-heading-icon"><TerminalSquare size={24} /></div>
        <div className="mcp-heading-copy">
          <span>READ-ONLY PROJECT BRIDGE</span>
          <h2>Ghidra MCP Investigator</h2>
          <p>Ask evidence-led questions of binaries already indexed inside the local Ghidra project.</p>
        </div>
        <div className="mcp-health">
          <div className="mcp-live-line"><StatusDot ok={status?.ok} /><strong>{statusBusy && !status ? 'CHECKING' : status?.ok ? 'LIVE' : 'OFFLINE'}</strong></div>
          <span>{status?.tool_count ?? '—'} tools · {status?.read_only_tools.length ?? '—'} read-only</span>
          {status?.server && <small>{status.server} {status.version ?? ''}</small>}
        </div>
        <button className="mcp-refresh" onClick={() => void refresh()} disabled={statusBusy} aria-label="Refresh MCP bridge status">
          <RefreshCw className={statusBusy ? 'spin' : ''} size={16} />
        </button>
      </div>

      <div className="mcp-guardrail">
        <div><ShieldCheck size={19} /><span><strong>{mutationCount} mutation tools blocked</strong><small>Evidence can be read; project state cannot be changed from this app.</small></span></div>
        <code>{status?.binding ?? 'loopback-only'}</code>
      </div>

      <div className="mcp-investigator-body">
        <div className="mcp-question-panel">
          <label className="field-label" htmlFor="mcp-binary">PROJECT BINARY</label>
          <div className="mcp-binary-select">
            <Binary size={17} />
            <select
              id="mcp-binary"
              data-testid="mcp-binary-select"
              value={binaryName}
              disabled={!status?.ok || binaries.length === 0}
              onChange={(event) => { setBinaryName(event.target.value); setResult(null); setInvestigationError('') }}
            >
              {binaries.length === 0 && <option value="">No indexed project binaries</option>}
              {binaries.map((binary) => (
                <option key={binary.name} value={binary.name}>
                  {binary.name} · {binary.code_indexed && binary.strings_indexed ? 'indexed' : binary.analysis_complete ? 'analyzed' : 'processing'}
                </option>
              ))}
            </select>
            <ChevronDown size={14} />
          </div>
          {selectedBinary && (
            <div className="mcp-binary-meta">
              <span><Check size={12} />{selectedBinary.analysis_complete ? 'Analysis complete' : 'Analysis pending'}</span>
              <span className={selectedBinary.code_indexed ? 'is-ready' : ''}>Code index</span>
              <span className={selectedBinary.strings_indexed ? 'is-ready' : ''}>String index</span>
            </div>
          )}

          <label className="field-label" htmlFor="mcp-question">DEFENSIVE QUESTION</label>
          <textarea
            id="mcp-question"
            data-testid="mcp-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={5}
            placeholder="Where is this protocol parsed, and what evidence supports that conclusion?"
          />

          <div className="mcp-investigate-actions">
            <ModelSelect model={model} setModel={setModel} catalog={catalog} testId="mcp-model-select" />
            <button
              className="primary-button"
              data-testid="mcp-investigate"
              onClick={() => void investigate()}
              disabled={!status?.ok || !binaryName || question.trim().length < 8 || !availableModel || investigating}
            >
              {investigating ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
              {investigating ? 'Reading evidence…' : 'Investigate'}
            </button>
          </div>
          <ModelGateNote catalog={catalog} kind="text" />
          {statusError && <div className="mcp-inline-state"><Activity size={15} /><span><strong>Bridge unavailable</strong>{statusError}</span></div>}
          {!statusError && status?.ok && binaries.length === 0 && <div className="mcp-inline-state"><Cpu size={15} /><span><strong>No indexed target yet</strong>Start the prepared PyGhidra-MCP workbench with a project binary.</span></div>}
          {investigationError && <div className="inline-error"><Activity size={15} />{investigationError}</div>}
        </div>

        <div className={`mcp-answer-panel ${result ? 'has-answer' : ''}`}>
          {result ? (
            <>
              <div className="mcp-answer-header">
                <div><span>READ-ONLY FINDINGS</span><h3>{binaryName}</h3></div>
                <div><Check size={13} />{Object.keys(result.evidence).length} evidence groups</div>
              </div>
              <div className="mcp-answer-markdown"><Markdown>{result.analysis}</Markdown></div>
              <div className="mcp-result-safety"><ShieldCheck size={15} /><span>{result.safety}</span></div>
            </>
          ) : (
            <div className="mcp-answer-empty">
              <div><Search size={24} /></div>
              <strong>{investigating ? 'Correlating Ghidra evidence…' : 'Your investigation will appear here'}</strong>
              <p>Findings cite project evidence and keep observations separate from hypotheses.</p>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function ReverseView({ catalog }: { catalog: CatalogResponse | null }) {
  const [toolchain, setToolchain] = useState<Record<string, Record<string, unknown>> | null>(null)
  const [toolchainError, setToolchainError] = useState('')
  const [metadata, setMetadata] = useState<BinaryMetadata | null>(null)
  const [analysis, setAnalysis] = useState('')
  const [workbenchError, setWorkbenchError] = useState('')
  const [busy, setBusy] = useState(false)
  const [model, setModel] = useState('localllm-deep')
  const [deleteArmed, setDeleteArmed] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const [deleteSuccess, setDeleteSuccess] = useState('')
  const availableModel = chooseAvailableAlias(catalog, model, 'text')
  const refreshToolchain = useCallback(async () => {
    try {
      setToolchain(await api.toolchain())
      setToolchainError('')
    } catch (reason) {
      setToolchainError(reason instanceof Error ? reason.message : String(reason))
    }
  }, [])
  useEffect(() => { void refreshToolchain() }, [refreshToolchain])
  useEffect(() => {
    if (availableModel && availableModel !== model) setModel(availableModel)
  }, [availableModel, model])
  useEffect(() => {
    if (!deleteArmed) return
    const timer = window.setTimeout(() => setDeleteArmed(false), 6000)
    return () => window.clearTimeout(timer)
  }, [deleteArmed])
  const inspect = async (file?: File) => {
    if (!file) return
    if (file.size > 64 * 1024 * 1024) {
      setWorkbenchError('This binary exceeds the 64 MB local inspection limit.')
      return
    }
    setBusy(true); setAnalysis(''); setWorkbenchError(''); setDeleteSuccess(''); setDeleteError(''); setDeleteArmed(false)
    try { setMetadata(await api.inspectBinary(file)) }
    catch (reason) { setWorkbenchError(reason instanceof Error ? reason.message : String(reason)) }
    finally { setBusy(false) }
  }
  const triage = async () => {
    if (!metadata || !availableModel) return
    setBusy(true); setWorkbenchError('')
    try { setAnalysis((await api.triageBinary(metadata, availableModel)).analysis) }
    catch (reason) { setWorkbenchError(reason instanceof Error ? reason.message : String(reason)) }
    finally { setBusy(false) }
  }
  const deleteLocalArtifact = async () => {
    if (!metadata || deleting) return
    if (!deleteArmed) {
      setDeleteArmed(true)
      setDeleteError('')
      return
    }
    const filename = metadata.filename
    setDeleteArmed(false)
    setDeleting(true)
    setDeleteError('')
    try {
      await api.deleteInspection(metadata.id)
      setMetadata(null)
      setAnalysis('')
      setDeleteSuccess(`${filename} and its inspection metadata were deleted from local storage.`)
    } catch (reason) {
      setDeleteError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setDeleting(false)
    }
  }
  const toolRows = [
    ['Ghidra 12.0.3', Boolean(toolchain?.ghidra?.installed), 'Decompiler + static analysis'],
    ['OGhidra', Boolean(toolchain?.oghidra?.installed), 'Local agent loop via Ollama'],
    ['PyGhidra MCP', Boolean(toolchain?.pyghidra_mcp?.installed), 'Headless project automation'],
    ['USB evidence', Boolean(toolchain?.usb?.evidence_container || toolchain?.usb?.tshark), 'Offline TShark / libusb lane'],
  ] as const
  return (
    <section className="padded-view reverse-view">
      <HeroTitle eyebrow="DEFENSIVE REVERSE ENGINEERING" title="From opaque binary" accent="to testable evidence." copy="Triage safely in the browser, then hand verified targets to Ghidra, MCP, packet captures, and your local 30B agent." />
      <div className="pipeline"><div><FileCode2 size={21} /><strong>Binary</strong><span>.sys · .dll · firmware</span></div><ArrowRight /><div><Search size={21} /><strong>Static evidence</strong><span>hash · strings · imports</span></div><ArrowRight /><div><BrainCircuit size={21} /><strong>Local model</strong><span>hypotheses · rename plan</span></div><ArrowRight /><div><ShieldCheck size={21} /><strong>Verify</strong><span>Ghidra · capture · hardware</span></div></div>
      <div className="reverse-layout">
        <div className="toolchain-panel"><div className="panel-title"><div><MonitorCog size={19} /><span><strong>Toolchain health</strong><small>LOCAL SERVICES</small></span></div><button onClick={() => void refreshToolchain()}><RefreshCw size={15} /></button></div>{toolRows.map(([name, ready, copy]) => <div className="tool-row" key={name}><StatusDot ok={ready} /><div><strong>{name}</strong><span>{copy}</span></div><small>{ready ? 'READY' : 'SETUP'}</small></div>)}{toolchainError && <div className="inline-error"><Activity size={15} />{toolchainError}</div>}<a className="docs-link" href="/references/reverse-engineering-workflow.md"><BookOpen size={15} /> Open operator guide <ArrowRight size={14} /></a></div>
        <div className="binary-workbench">
          <div className="panel-title"><div><Binary size={19} /><span><strong>Safe binary triage</strong><small>NEVER EXECUTES UPLOADS</small></span></div></div>
          {!metadata ? (
            <>
              {deleteSuccess && <div className="binary-delete-success" role="status"><Check size={15} /><span><strong>Local artifact deleted</strong>{deleteSuccess}</span></div>}
              <label className="binary-drop">
                <input hidden type="file" onChange={(event) => void inspect(event.target.files?.[0])} />
                {busy ? <LoaderCircle className="spin" size={28} /> : <UploadCloud size={28} />}
                <strong>{busy ? 'Inspecting metadata…' : 'Choose a binary to inspect'}</strong>
                <p>Up to 64 MB · stored locally · static inspection only</p>
              </label>
            </>
          ) : (
            <div className="binary-result">
              <div className="binary-name">
                <div><FileCode2 size={21} /></div>
                <span><strong>{metadata.filename}</strong><small>{metadata.file_type}</small></span>
                <small className="binary-stored-badge"><HardDrive size={12} /> STORED LOCALLY</small>
              </div>
              <div className="hash-row"><span>SHA-256</span><code>{metadata.sha256}</code></div>
              <div className="binary-stats"><span><strong>{formatBytes(metadata.size)}</strong> size</span><span><strong>{metadata.strings.length}</strong> strings sampled</span></div>
              <div className="binary-agent-actions"><ModelSelect model={model} setModel={setModel} catalog={catalog} /><button className="primary-button" onClick={() => void triage()} disabled={busy || deleting || !availableModel}>{busy ? <LoaderCircle className="spin" size={16} /> : <BrainCircuit size={16} />} Ask local RE agent</button></div>
              <ModelGateNote catalog={catalog} kind="text" />
              <div className={`binary-retention ${deleteArmed ? 'is-armed' : ''}`}>
                <div><ShieldCheck size={16} /><span><strong>Local retention</strong><small>The uploaded binary and JSON metadata remain only on this machine.</small></span></div>
                <div className="binary-delete-actions">
                  {deleteArmed && <button className="cancel-delete" onClick={() => setDeleteArmed(false)} disabled={deleting}>Cancel</button>}
                  <button
                    className="delete-artifact-button"
                    data-testid="delete-local-artifact"
                    data-confirmation={deleteArmed ? 'armed' : 'idle'}
                    onClick={() => void deleteLocalArtifact()}
                    disabled={deleting || busy}
                  >
                    {deleting ? <LoaderCircle className="spin" size={14} /> : <Trash2 size={14} />}
                    {deleting ? 'Deleting…' : deleteArmed ? 'Confirm permanent delete' : 'Delete local artifact'}
                  </button>
                </div>
              </div>
              {deleteArmed && <p className="delete-confirm-copy" role="alert">Click confirm to remove both the stored binary and its inspection metadata. This cannot be undone.</p>}
              {deleteError && <div className="inline-error"><Activity size={15} />{deleteError}</div>}
            </div>
          )}
          {workbenchError && <div className="inline-error"><Activity size={15} />{workbenchError}</div>}
        </div>
      </div>
      {analysis && <div className="triage-report"><div className="report-header"><div><span>LOCAL AGENT NOTES</span><h2>Evidence-led triage</h2></div></div><Markdown>{analysis}</Markdown></div>}
      <McpInvestigator catalog={catalog} model={model} setModel={setModel} />
      <div className="safety-banner"><ShieldCheck size={22} /><div><strong>Binary content is untrusted data.</strong><p>The agent is explicitly instructed to ignore embedded prompt-like strings. Its conclusions remain hypotheses until supported by cross-references, packet captures, tests, or hardware behavior.</p></div></div>
    </section>
  )
}

function CodeBlock({ title, code }: { title: string; code: string }) {
  const [copied, setCopied] = useState(false)
  return <div className="code-card"><div><span>{title}</span><button onClick={() => { void navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1200) }}>{copied ? <Check size={14} /> : <Clipboard size={14} />}{copied ? 'Copied' : 'Copy'}</button></div><pre><code>{code}</code></pre></div>
}

function ApiView() {
  const python = `from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8008/v1",
    api_key="local-dev-key",
)

response = client.responses.create(
    model="localllm-deep",
    input="Design a fault-tolerant robot control loop.",
)
print(response.output_text)`
  const curl = `curl http://127.0.0.1:8008/v1/chat/completions \\
  -H "Authorization: Bearer local-dev-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "localllm-fast",
    "messages": [{"role": "user", "content": "Hello locally!"}],
    "stream": true
  }'`
  const vision = `client.chat.completions.create(
    model="localllm-vision",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Inspect this diagram"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }],
)`
  return (
    <section className="padded-view api-view">
      <HeroTitle eyebrow="DROP-IN LOCAL API" title="Familiar format." accent="Private backend." copy="Point OpenAI SDKs and compatible tools at one loopback URL. Friendly aliases keep your applications independent from exact quantization tags." />
      <div className="endpoint-hero"><div className="endpoint-icon"><Server size={25} /></div><div><span>BASE URL</span><code>http://127.0.0.1:8008/v1</code></div><button onClick={() => navigator.clipboard.writeText('http://127.0.0.1:8008/v1')}><Clipboard size={17} /></button><i>LOCAL</i></div>
      <div className="api-grid"><CodeBlock title="RESPONSES API · PYTHON" code={python} /><CodeBlock title="CHAT COMPLETIONS · CURL" code={curl} /><CodeBlock title="VISION INPUT · PYTHON" code={vision} /><div className="alias-card"><div className="panel-kicker"><Layers3 size={17} /><span>STABLE MODEL ALIASES</span></div>{[['localllm-pocket', 'Qwen3 4B Q4'], ['localllm-fast', 'Qwen3 8B Q4'], ['localllm-deep', 'Qwen3 30B Q4'], ['localllm-max', 'Qwen3 30B Q8'], ['localllm-vision', 'Qwen3-VL 8B Q4'], ['localllm-embed', 'BGE-M3 embeddings']].map(([alias, target]) => <div key={alias}><code>{alias}</code><ArrowRight size={13} /><span>{target}</span></div>)}</div></div>
      <div className="api-features"><div><MessageCircleMore /><strong>Chat Completions</strong><code>POST /v1/chat/completions</code></div><div><Sparkles /><strong>Responses</strong><code>POST /v1/responses</code></div><div><Boxes /><strong>Models</strong><code>GET /v1/models</code></div><div><BrainCircuit /><strong>Embeddings</strong><code>POST /v1/embeddings</code></div></div>
      <div className="api-note"><KeyRound size={21} /><div><strong>Authentication that local tools understand</strong><p>Use <code>local-dev-key</code> by default or set <code>LOCALLLM_API_KEY</code>. LocalLLM enforces a loopback-only boundary; use an authenticated SSH or VPN tunnel for remote access.</p></div></div>
    </section>
  )
}

function App() {
  const [view, setView] = useState<ViewId>('chat')
  const [mobileNav, setMobileNav] = useState(false)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null)

  const refreshStatus = useCallback(() => { void api.system().then(setStatus).catch(() => setStatus(null)) }, [])
  const refreshCatalog = useCallback(async () => { try { setCatalog(await api.catalog()) } catch { setCatalog(null) } }, [])
  useEffect(() => {
    refreshStatus()
    void refreshCatalog()
    const timer = window.setInterval(() => {
      refreshStatus()
      void refreshCatalog()
    }, 15000)
    return () => window.clearInterval(timer)
  }, [refreshCatalog, refreshStatus])

  return (
    <div className="app-shell" data-testid="workspace" data-status="ready" data-view={view}>
      <Sidebar view={view} setView={setView} open={mobileNav} setOpen={setMobileNav} />
      <main className="main-shell">
        <AppHeader status={status} onRefresh={refreshStatus} />
        <div className="workspace-view" data-testid="view-chat-panel" hidden={view !== 'chat'}><ChatView catalog={catalog} /></div>
        <div className="workspace-view" data-testid="view-vision-panel" hidden={view !== 'vision'}><VisionView catalog={catalog} /></div>
        <div className="workspace-view" data-testid="view-research-panel" hidden={view !== 'research'}><ResearchView catalog={catalog} /></div>
        <div className="workspace-view" data-testid="view-models-panel" hidden={view !== 'models'}><ModelsView catalog={catalog} refresh={refreshCatalog} /></div>
        <div className="workspace-view" data-testid="view-reverse-panel" hidden={view !== 'reverse'}><ReverseView catalog={catalog} /></div>
        <div className="workspace-view" data-testid="view-api-panel" hidden={view !== 'api'}><ApiView /></div>
      </main>
    </div>
  )
}

export default App

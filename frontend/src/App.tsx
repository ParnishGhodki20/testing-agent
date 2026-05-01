import { useState, useRef, useEffect, useCallback } from 'react'
import type { DragEvent, ChangeEvent, KeyboardEvent } from 'react'
import { ChainlitClient } from './chainlit'
import type { StepDict, AskPayload, ConnectionStatus, FileRef } from './chainlit'
import './App.css'
import ReactMarkdown from 'react-markdown'

type Role = 'user' | 'assistant'
interface Message { id: string; role: Role; content: string; isTyping?: boolean; stepLabel?: string; streaming?: boolean; ask?: AskPayload }
interface ChatSession { id: string; title: string; messages: Message[]; createdAt: Date }
interface UploadedFile { file: File; id: string }

const IconMenu = () => (<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>)
const IconEdit = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>)
const IconSearch = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>)

const IconCode = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>)
const IconMore = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/></svg>)
const IconChevronDown = () => (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"/></svg>)
const IconSend = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>)
const IconMic = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>)
const IconPaperclip = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>)
const IconPlus = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>)
const IconCopy = () => (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>)
const IconThumbUp = () => (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>)
const IconThumbDown = () => (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>)
const IconRefresh = () => (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>)
const IconFile = () => (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>)
const IconUpload = () => (<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>)
const IconShare = () => (<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>)
const IconWifi = () => (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>)
const IconTrash = () => (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>)

function uid() { return Math.random().toString(36).slice(2) }
function formatBytes(b: number) { return b < 1024 ? b + ' B' : b < 1048576 ? (b/1024).toFixed(1)+' KB' : (b/1048576).toFixed(1)+' MB' }

const STATUS_COLOR: Record<ConnectionStatus, string> = { disconnected:'#6b6b6b', connecting:'#f59e0b', connected:'#10a37f', error:'#ef4444' }
const STATUS_LABEL: Record<ConnectionStatus, string> = { disconnected:'Disconnected', connecting:'Connecting…', connected:'Connected', error:'Connection error' }
const CHAINLIT_URL = window.location.origin === 'http://localhost:3000' 
  ? 'http://localhost:8000' 
  : window.location.origin


// ── Sidebar ──────────────────────────────────────────────────────────────────
function Sidebar({ collapsed, sessions, activeId, status, onNewChat, onSelectSession, onDeleteSession }: {
  collapsed: boolean; sessions: ChatSession[]; activeId: string; status: ConnectionStatus
  onNewChat: () => void; onSelectSession: (id: string) => void; onDeleteSession: (id: string) => void
}) {
  const [searchQuery, setSearchQuery] = useState('');
  
  const filteredSessions = sessions.filter(s => 
    s.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <aside className={`sidebar${collapsed ? ' collapsed' : ''}`}>
      <div className="sidebar-header">
        <img src="/public/logo.png" alt="Icertis Logo" className="sidebar-logo-img" />
        <span className="sidebar-logo">Icertis</span>
        <button className="sidebar-icon-btn" onClick={onNewChat} title="New chat"><IconEdit /></button>
      </div>
      <button className="new-chat-btn" onClick={onNewChat}><IconPlus /> New chat</button>
      
      <div className="sidebar-search-container">
        <IconSearch />
        <input 
          type="text" 
          placeholder="Search chats" 
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          className="sidebar-search-input"
        />
      </div>

      {sessions.length > 0 && (<>
        <div className="sidebar-section-label">Recents</div>
        <div className="sidebar-history">
          {filteredSessions.length > 0 ? filteredSessions.map(s => (
            <div key={s.id} className={`history-item-container${s.id === activeId ? ' active' : ''}`}>
              <button className="history-item" onClick={() => onSelectSession(s.id)} title={s.title}>
                {s.title}
              </button>
              <button className="history-delete-btn" onClick={(e) => { e.stopPropagation(); onDeleteSession(s.id); }} title="Delete chat">
                <IconTrash />
              </button>
            </div>
          )) : (
            <div className="sidebar-empty-state">No matching chats</div>
          )}
        </div>
      </>)}
      <div className="sidebar-footer">
        <div className="conn-status">
          <span style={{ color: STATUS_COLOR[status] }}><IconWifi /></span>
          <span style={{ fontSize: 12, color: STATUS_COLOR[status] }}>{STATUS_LABEL[status]}</span>
        </div>
        <button className="user-profile">
          <div className="user-avatar">T</div>
          <span className="user-name">Testing Agent</span>
        </button>
      </div>
    </aside>
  )
}

// ── Upload Modal ──────────────────────────────────────────────────────────────
function UploadModal({ onClose, onConfirm, uploading, uploadProgress }: {
  onClose: () => void; onConfirm: (files: File[]) => void; uploading: boolean; uploadProgress: number
}) {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const addFiles = (fl: FileList | null) => { if (!fl) return; setFiles(p => [...p, ...Array.from(fl).map(f => ({ file: f, id: uid() }))]) }
  const onDrop = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setDragOver(false); addFiles(e.dataTransfer.files) }
  return (
    <div className="upload-overlay" onClick={e => e.target === e.currentTarget && !uploading && onClose()}>
      <div className="upload-modal">
        <h2>Upload documents</h2>
        <p>Upload requirement or specification documents. Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT, JSON, XML, images, ZIP.</p>
        <div className={`upload-drop-zone${dragOver ? ' drag-over' : ''}`}
          onClick={() => !uploading && inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)} onDrop={onDrop}>
          <IconUpload />
          <span className="upload-drop-text">Click to browse or drag files here</span>
          <span className="upload-drop-hint">PDF, DOCX, PPTX, XLSX, CSV, TXT, JSON, XML, images, ZIP</span>
          <input ref={inputRef} type="file" multiple style={{ display:'none' }}
            onChange={(e: ChangeEvent<HTMLInputElement>) => addFiles(e.target.files)} />
        </div>
        {files.length > 0 && (
          <div className="upload-file-list">
            {files.map(f => (
              <div key={f.id} className="upload-file-item">
                <IconFile /><span className="upload-file-name">{f.file.name}</span>
                <span className="upload-file-size">{formatBytes(f.file.size)}</span>
              </div>
            ))}
          </div>
        )}
        {uploading && (<div className="upload-progress-bar"><div className="upload-progress-fill" style={{ width:`${uploadProgress}%` }} /></div>)}
        <div className="upload-actions">
          <button className="btn-secondary" onClick={onClose} disabled={uploading}>Cancel</button>
          <button className="btn-primary" disabled={files.length === 0 || uploading}
            onClick={() => onConfirm(files.map(f => f.file))}>
            {uploading ? `Uploading… ${uploadProgress}%` : 'Start session'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Inline Upload ──
function InlineUpload({ ask, onUpload, uploading, uploadProgress }: {
  ask: AskPayload; onUpload: (files: File[]) => void; uploading: boolean; uploadProgress: number
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])

  const onSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fl = Array.from(e.target.files)
      setFiles(fl)
      onUpload(fl)
    }
  }

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    if (e.dataTransfer.files) {
      const fl = Array.from(e.dataTransfer.files)
      setFiles(fl)
      onUpload(fl)
    }
  }

  return (
    <div className="inline-upload" onDragOver={e => e.preventDefault()} onDrop={onDrop}>
      <div className="inline-upload-info">
        <span className="inline-upload-title">Drag and drop files here</span>
        <span className="inline-upload-hint">Limit: {ask.spec.max_size_mb || 500}mb</span>
        {files.length > 0 && (
          <div className="inline-file-list">
            {files.map((f, i) => (
              <div key={i} className="inline-file-item">
                <IconFile /> {f.name}
              </div>
            ))}
          </div>
        )}
        {uploading && (
          <div className="upload-progress-bar" style={{ marginTop: 8 }}>
            <div className="upload-progress-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
        )}
      </div>
      <button className="btn-pink" onClick={() => inputRef.current?.click()} disabled={uploading}>
        <IconUpload /> {uploading ? 'Uploading...' : 'Browse Files'}
      </button>
      <input ref={inputRef} type="file" multiple style={{ display: 'none' }} onChange={onSelect} />
    </div>
  )
}

// ── Message ───────────────────────────────────────────────────────────────────
function MessageItem({ msg, onUpload, uploading, uploadProgress }: { 
  msg: Message; onUpload?: (files: File[]) => void; uploading?: boolean; uploadProgress?: number 
}) {
  const [copied, setCopied] = useState(false)
  const copy = () => { navigator.clipboard.writeText(msg.content); setCopied(true); setTimeout(() => setCopied(false), 1500) }
  return (
    <div className={`message-row ${msg.role}`}>
      <div className="message-inner">
        {msg.role === 'assistant' && <div className="message-avatar assistant-avatar">TC</div>}
        <div className="message-content">
          {msg.isTyping ? (
            <div className="message-bubble">
              {msg.stepLabel
                ? <div className="step-indicator"><div className="step-spinner" />{msg.stepLabel}</div>
                : <div className="typing-indicator"><div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" /></div>}
            </div>
          ) : (
            <>
              <div className="message-bubble">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
                {msg.ask && onUpload && (
                  <InlineUpload ask={msg.ask} onUpload={onUpload} uploading={uploading || false} uploadProgress={uploadProgress || 0} />
                )}
              </div>
              {msg.role === 'assistant' && !msg.streaming && (
                <div className="message-actions">
                  <button className="msg-action-btn" onClick={copy} title="Copy"><IconCopy /></button>
                  <button className="msg-action-btn" title="Good response"><IconThumbUp /></button>
                  <button className="msg-action-btn" title="Bad response"><IconThumbDown /></button>
                  <button className="msg-action-btn" title="Regenerate"><IconRefresh /></button>
                  {copied && <span style={{ fontSize:12, color:'var(--text-muted)', alignSelf:'center' }}>Copied!</span>}
                </div>
              )}
            </>
          )}
        </div>
        {msg.role === 'user' && <div className="message-avatar user-avatar-msg">T</div>}
      </div>
    </div>
  )
}

const CHIPS = [
  { icon: <IconFile />,   label: 'Generate scenarios' },
  { icon: <IconCode />,   label: 'Generate test cases' },
  { icon: <IconSearch />, label: 'Look something up' },
]

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [sidebarOpen, setSidebarOpen]       = useState(true)
  const [sessions, setSessions]             = useState<ChatSession[]>([])
  const [activeId, setActiveId]             = useState<string>('')
  const [input, setInput]                   = useState('')
  const [isLoading, setIsLoading]           = useState(false)
  const [showUpload, setShowUpload]         = useState(false)
  const [uploading, setUploading]           = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [status, setStatus]                 = useState<ConnectionStatus>('disconnected')
  const [activeAskId, setActiveAskId]       = useState<string | null>(null)
  const [shareCopied, setShareCopied]       = useState(false)

  const pendingAskRef = useRef<((answer: unknown) => void) | null>(null)
  const textareaRef   = useRef<HTMLTextAreaElement>(null)
  const bottomRef     = useRef<HTMLDivElement>(null)
  const clientRef     = useRef<ChainlitClient | null>(null)
  const activeIdRef   = useRef<string>('')

  useEffect(() => { activeIdRef.current = activeId }, [activeId])

  const activeSession = sessions.find(s => s.id === activeId) ?? null

  const createSession = useCallback((title = 'New chat') => {
    const id = uid()
    setSessions(prev => [{ id, title, messages: [], createdAt: new Date() }, ...prev])
    setActiveId(id); activeIdRef.current = id
    return id
  }, [])

  const addMessage = useCallback((sessionId: string, msg: Message) => {
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, messages: [...s.messages, msg] } : s))
  }, [])

  const appendToken = useCallback((sessionId: string, msgId: string, token: string) => {
    setSessions(prev => prev.map(s => s.id !== sessionId ? s : {
      ...s, messages: s.messages.map(m => m.id === msgId ? { ...m, content: m.content + token, streaming: true } : m)
    }))
  }, [])

  const updateTitle = useCallback((sessionId: string, title: string) => {
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, title } : s))
  }, [])

  const handleDeleteSession = useCallback((idToDelete: string) => {
    setSessions(prev => {
      const updated = prev.filter(s => s.id !== idToDelete)
      if (activeIdRef.current === idToDelete) {
        setActiveId(updated.length > 0 ? updated[0].id : '')
      }
      return updated
    })
  }, [])

  useEffect(() => {
    const ta = textareaRef.current; if (!ta) return
    ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
  }, [input])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [activeSession?.messages])

  // ── Connect to Chainlit ────────────────────────────────────────────────────
  useEffect(() => {
    createSession('New chat')

    const client = new ChainlitClient(CHAINLIT_URL, {
      onConnectionChange: setStatus,
      onTaskStart: () => setIsLoading(true),
      onTaskEnd:   () => setIsLoading(false),

      onNewMessage: (step: StepDict) => {
        if (step.type !== 'assistant_message' && step.type !== 'message') return

        const sid = activeIdRef.current
        setSessions(prev => prev.map(s => {
          if (s.id !== sid) return s
          const typingIdx = s.messages.findIndex(m => m.isTyping)
          const newMsg: Message = { id: step.id, role: 'assistant', content: step.output || '', streaming: false }
          if (typingIdx !== -1) { const msgs = [...s.messages]; msgs[typingIdx] = newMsg; return { ...s, messages: msgs } }
          return { ...s, messages: [...s.messages, newMsg] }
        }))
      },

      onUpdateMessage: (step: StepDict) => {
        if (step.type !== 'assistant_message' && step.type !== 'message') return
        const sid = activeIdRef.current

        setSessions(prev => prev.map(s => s.id !== sid ? s : {
          ...s, messages: s.messages.map(m => m.id === step.id
            ? { ...m, content: step.output || m.content, streaming: false } : m)
        }))
      },

      onStreamStart: (step: StepDict) => {
        if (step.type !== 'assistant_message' && step.type !== 'message') return
        const sid = activeIdRef.current

        setSessions(prev => prev.map(s => s.id !== sid ? s : {
          ...s, messages: [...s.messages.filter(m => !m.isTyping),
            { id: step.id, role: 'assistant' as Role, content: '', streaming: true }]
        }))
      },

      onStreamToken: (id: string, token: string) => appendToken(activeIdRef.current, id, token),

      onAsk: (payload: AskPayload, respond) => {
        pendingAskRef.current = respond
        setActiveAskId(payload.msg.id)
        
        const sid = activeIdRef.current
        setSessions(prev => prev.map(s => {
          if (s.id !== sid) return s
          const newMsg: Message = { 
            id: payload.msg.id, 
            role: 'assistant', 
            content: payload.msg.output || 'Please upload your documents to begin.',
            ask: payload
          }
          // Replace typing indicator if exists
          const typingIdx = s.messages.findIndex(m => m.isTyping)
          if (typingIdx !== -1) { 
            const msgs = [...s.messages]; msgs[typingIdx] = newMsg; return { ...s, messages: msgs } 
          }
          return { ...s, messages: [...s.messages, newMsg] }
        }))
      },


      onError: (msg: string) => {
        addMessage(activeIdRef.current, { id: uid(), role: 'assistant', content: `⚠️ ${msg}` })
        setIsLoading(false)
      },
    })

    clientRef.current = client
    client.connect()
    return () => { client.disconnect() }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Send ───────────────────────────────────────────────────────────────────
  const handleSend = useCallback(() => {
    const text = input.trim()
    if (!text || isLoading) return
    let sid = activeIdRef.current
    if (!sid) sid = createSession(text.slice(0, 40))
    else if (activeSession?.messages.length === 0) updateTitle(sid, text.slice(0, 40))

    addMessage(sid, { id: uid(), role: 'user', content: text })
    setInput('')

    if (pendingAskRef.current) {
      const respond = pendingAskRef.current; pendingAskRef.current = null
      respond({ id: uid(), output: text, type: 'user_message', createdAt: new Date().toISOString() })
    } else {
      addMessage(sid, { id: uid(), role: 'assistant', content: '', isTyping: true, stepLabel: 'Thinking…' })
      clientRef.current?.sendMessage(text)
    }
  }, [input, isLoading, activeSession, createSession, updateTitle, addMessage])

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  const handleNewChat = () => { clientRef.current?.newSession(); createSession('New chat') }

  // ── Upload ─────────────────────────────────────────────────────────────────
  const handleUploadConfirm = async (files: File[]) => {
    if (!clientRef.current) return
    setUploading(true); setUploadProgress(0)
    try {
      const refs: FileRef[] = []
      for (let i = 0; i < files.length; i++) {
        const ref = await clientRef.current.uploadFile(files[i], clientRef.current.currentSessionId,
          pct => setUploadProgress(Math.round((i / files.length) * 100 + pct / files.length)))
        refs.push(ref)
      }
      setUploadProgress(100)
      
      const rawName = files[0].name
      const cleanName = rawName.replace(/\.[^/.]+$/, "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
      
      const sid = activeIdRef.current
      if (!activeSession || activeSession.title === 'New chat') {
        updateTitle(sid, cleanName)
        addMessage(sid, { id: uid(), role: 'user', content: `📄 ${cleanName}` })
      }
      
      if (pendingAskRef.current) {
        const respond = pendingAskRef.current; pendingAskRef.current = null
        respond(refs.map(r => ({ id: r.id })))
      }
      setActiveAskId(null)
      setShowUpload(false)
    } catch (err) {
      addMessage(activeIdRef.current, { id: uid(), role: 'assistant', content: `⚠️ Upload failed: ${err}` })
    } finally { setUploading(false); setUploadProgress(0) }
  }

  const handleShare = () => {
    if (!activeSession || activeSession.messages.length === 0) return
    let text = `# ${activeSession.title}\n\n`
    let hasContent = false
    
    activeSession.messages.forEach(m => {
      if (m.isTyping || !m.content) return
      // Skip the default welcome message
      if (m.content.includes('Upload one or more feature / requirement documents to begin.')) return
      
      const role = m.role === 'user' ? 'User' : 'Testing Copilot'
      text += `**${role}**:\n${m.content}\n\n`
      hasContent = true
    })
    
    if (!hasContent) {
      alert("No test cases to share yet! Generate some test cases first.")
      return
    }

    // Create a blob and trigger download
    const blob = new Blob([text], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${activeSession.title.replace(/\s+/g, '_')}_Test_Cases.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    setShareCopied(true)
    setTimeout(() => setShareCopied(false), 2000)
  }

  const showWelcome = !activeSession || activeSession.messages.length === 0

  return (
    <>
      <Sidebar collapsed={!sidebarOpen} sessions={sessions} activeId={activeId} status={status}
        onNewChat={handleNewChat} onSelectSession={id => setActiveId(id)} onDeleteSession={handleDeleteSession} />

      <div className="main">
        <div className="topbar">
          <button className="topbar-toggle" onClick={() => setSidebarOpen(o => !o)} title="Toggle sidebar"><IconMenu /></button>
          <button className="model-selector">Testing Copilot <IconChevronDown /></button>
          <div className="topbar-actions">
            <button 
              className="topbar-action-btn" 
              title="Download Markdown" 
              onClick={handleShare}
              style={shareCopied ? { width: 'auto', padding: '0 12px' } : undefined}
            >
              {shareCopied ? <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>Downloaded!</span> : <IconShare />}
            </button>
            <button className="topbar-action-btn" title="More options"><IconMore /></button>
          </div>
        </div>

        <div className="chat-area">
          {showWelcome ? (
            <div className="welcome-screen"><h1 className="welcome-title">Ready when you are.</h1></div>
          ) : (
          <div className="messages-container">
              {activeSession!.messages.map(msg => (
                <MessageItem 
                  key={msg.id} 
                  msg={msg} 
                  onUpload={msg.id === activeAskId ? handleUploadConfirm : undefined}
                  uploading={msg.id === activeAskId ? uploading : false}
                  uploadProgress={msg.id === activeAskId ? uploadProgress : 0}
                />
              ))}
              <div ref={bottomRef} />
            </div>

          )}
        </div>

        <div className="input-area">
          <div className="input-wrapper">
            <div className="input-top">
              <textarea ref={textareaRef} className="chat-input" placeholder="Ask anything"
                value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
                rows={1} disabled={isLoading || status === 'connecting'} />
            </div>
            <div className="input-bottom">
              <div className="input-left-actions">
                <button className="input-action-btn" title="Attach file" onClick={() => setShowUpload(true)}><IconPaperclip /></button>
                <button className="input-action-btn" title="More tools"><IconPlus /></button>
              </div>
              <div style={{ display:'flex', gap:6, alignItems:'center' }}>
                <button className="input-action-btn" title="Voice input"><IconMic /></button>
                <button className="send-btn" onClick={handleSend}
                  disabled={!input.trim() || isLoading || status !== 'connected'} title="Send">
                  <IconSend />
                </button>
              </div>
            </div>
          </div>
          {showWelcome && (
            <div className="quick-chips">
              {CHIPS.map(c => (
                <button key={c.label} className="chip" onClick={() => { setInput(c.label); textareaRef.current?.focus() }}>
                  {c.icon}{c.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {showUpload && (
        <UploadModal onClose={() => { if (!uploading) setShowUpload(false) }}
          onConfirm={handleUploadConfirm} uploading={uploading} uploadProgress={uploadProgress} />
      )}
    </>
  )
}

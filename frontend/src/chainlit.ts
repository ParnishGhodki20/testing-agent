/**
 * Chainlit Socket.IO client
 *
 * Protocol reverse-engineered from:
 *   backend/chainlit/socket.py   — events the server listens for
 *   backend/chainlit/emitter.py  — events the server emits
 *
 * Flow:
 *   1. connect  (with auth payload)
 *   2. emit "connection_successful"
 *   3. server calls on_chat_start → emits "ask" (file upload prompt)
 *   4. we respond to "ask" with uploaded file refs
 *   5. user sends messages via "client_message"
 *   6. server streams back via "new_message" / "stream_start" / "stream_token" / "update_message"
 */

import { io, Socket } from 'socket.io-client'
import { v4 as uuidv4 } from 'uuid'

// ── Types ────────────────────────────────────────────────────────────────────

export interface StepDict {
  id: string
  name: string
  type: string
  output: string
  input?: string
  parentId?: string
  threadId?: string
  createdAt?: string
  isError?: boolean
  metadata?: Record<string, unknown>
  streaming?: boolean
}

export interface AskSpec {
  type: 'text' | 'file' | 'action'
  timeout: number
  accept?: Record<string, string[]>
  max_size_mb?: number
  max_files?: number
}

export interface AskPayload {
  msg: StepDict
  spec: AskSpec
}

export interface FileRef {
  id: string
  name: string
  type: string
  path?: string
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface ChainlitCallbacks {
  onConnectionChange: (status: ConnectionStatus) => void
  onNewMessage: (step: StepDict) => void
  onUpdateMessage: (step: StepDict) => void
  onStreamStart: (step: StepDict) => void
  onStreamToken: (id: string, token: string, isSequence: boolean) => void
  onTaskStart: () => void
  onTaskEnd: () => void
  onAsk: (payload: AskPayload, respond: (answer: unknown) => void) => void
  onError: (msg: string) => void
}

// ── Client class ─────────────────────────────────────────────────────────────

export class ChainlitClient {
  private socket: Socket | null = null
  private sessionId: string = uuidv4()
  private callbacks: ChainlitCallbacks
  private serverUrl: string

  constructor(serverUrl: string, callbacks: ChainlitCallbacks) {
    this.serverUrl = serverUrl
    this.callbacks = callbacks
  }

  connect() {
    if (this.socket?.connected) return

    this.callbacks.onConnectionChange('connecting')

    this.socket = io(this.serverUrl, {
      path: '/ws/socket.io',
      transports: ['websocket', 'polling'],
      auth: {
        sessionId: this.sessionId,
        clientType: 'webapp',
        userEnv: null,
        chatProfile: null,
        threadId: null,
      },


    })

    this.socket.on('connect', () => {
      this.callbacks.onConnectionChange('connected')
      // Required handshake — triggers on_chat_start
      this.socket!.emit('connection_successful')
    })

    this.socket.on('disconnect', () => {
      this.callbacks.onConnectionChange('disconnected')
    })

    this.socket.on('connect_error', (err) => {
      console.error('[Chainlit] connect_error', err)
      this.callbacks.onConnectionChange('error')
      this.callbacks.onError(`Connection failed: ${err.message}`)
    })

    // ── Server → Client events ──────────────────────────────────────────────

    this.socket.on('new_message', (step: StepDict) => {
      this.callbacks.onNewMessage(step)
    })

    this.socket.on('update_message', (step: StepDict) => {
      this.callbacks.onUpdateMessage(step)
    })

    this.socket.on('stream_start', (step: StepDict) => {
      this.callbacks.onStreamStart(step)
    })

    this.socket.on('stream_token', (data: { id: string; token: string; isSequence: boolean }) => {
      this.callbacks.onStreamToken(data.id, data.token, data.isSequence)
    })

    this.socket.on('task_start', () => {
      this.callbacks.onTaskStart()
    })

    this.socket.on('task_end', () => {
      this.callbacks.onTaskEnd()
    })

    // "ask" is a Socket.IO call — server waits for our ack response
    this.socket.on('ask', (payload: AskPayload, ack: (answer: unknown) => void) => {
      this.callbacks.onAsk(payload, ack)
    })

    this.socket.on('error', (msg: string) => {
      this.callbacks.onError(msg)
    })
  }

  disconnect() {
    this.socket?.disconnect()
    this.socket = null
  }

  /** Send a text message to the backend */
  sendMessage(content: string, fileRefs?: FileRef[]) {
    if (!this.socket?.connected) {
      this.callbacks.onError('Not connected to backend.')
      return
    }

    const payload = {
      message: {
        id: uuidv4(),
        output: content,
        type: 'user_message',
        createdAt: new Date().toISOString(),
      },
      fileReferences: fileRefs ?? null,
    }

    this.socket.emit('client_message', payload)
  }

  /** Upload a file to Chainlit's /project/file endpoint, returns a FileRef */
  async uploadFile(
    file: File,
    sessionId: string,
    onProgress?: (pct: number) => void
  ): Promise<FileRef> {
    return new Promise((resolve, reject) => {
      const formData = new FormData()
      formData.append('file', file)

      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${this.serverUrl}/project/file?session_id=${sessionId}`)

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      }

      xhr.onload = () => {
        if (xhr.status === 200) {
          const data = JSON.parse(xhr.responseText)
          resolve({
            id: data.id,
            name: file.name,
            type: file.type,
          })
        } else {
          reject(new Error(`Upload failed: ${xhr.status}`))
        }
      }

      xhr.onerror = () => reject(new Error('Upload network error'))
      xhr.send(formData)
    })
  }

  get currentSessionId() {
    return this.sessionId
  }

  get isConnected() {
    return this.socket?.connected ?? false
  }

  /** Reset session (new chat) */
  newSession() {
    this.socket?.emit('clear_session')
    this.disconnect()
    this.sessionId = uuidv4()
    this.connect()
  }
}

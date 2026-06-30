/**
 * WebSocket client for CRDT collaboration server.
 */

import { EventEmitter } from 'eventemitter3';
import {
  ClientId,
  DocumentId,
  ClientConfig,
  ClientEvents,
  ConnectionState,
  ServerMessage,
  ClientMessage,
  Operation,
  PresenceState,
  CursorPosition,
  Selection,
  DocumentSnapshot,
} from './types';
import { CRDTDocument, generateUUID } from './crdt';

/** Default client configuration. */
const DEFAULT_CONFIG: Partial<ClientConfig> = {
  maxReconnectAttempts: 5,
  reconnectDelay: 1000,
  heartbeatInterval: 30000,
};

/** Random color generator for presence. */
function generateColor(): string {
  const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
    '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1',
  ];
  return colors[Math.floor(Math.random() * colors.length)];
}

/** Collaboration client for real-time document editing. */
export class CollaborationClient extends EventEmitter<ClientEvents> {
  private config: ClientConfig;
  private ws: WebSocket | null = null;
  private state: ConnectionState = 'disconnected';
  private reconnectAttempts: number = 0;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private pendingMessages: ClientMessage[] = [];
  private documents: Map<DocumentId, CRDTDocument> = new Map();
  private currentDocId: DocumentId | null = null;
  private presence: Map<ClientId, PresenceState> = new Map();
  private operationSeq: number = 0;
  private pendingAcks: Set<number> = new Set();

  constructor(config: ClientConfig) {
    super();
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      clientId: config.clientId || generateUUID(),
      userColor: config.userColor || generateColor(),
    };
  }

  /** Get client ID. */
  get clientId(): ClientId {
    return this.config.clientId!;
  }

  /** Get current connection state. */
  get connectionState(): ConnectionState {
    return this.state;
  }

  /** Get current document. */
  get currentDocument(): CRDTDocument | null {
    return this.currentDocId ? this.documents.get(this.currentDocId) || null : null;
  }

  /** Get all users in the current document. */
  get users(): PresenceState[] {
    return Array.from(this.presence.values());
  }

  /** Connect to the collaboration server. */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.state === 'connected') {
        resolve();
        return;
      }

      this.setState('connecting');

      try {
        this.ws = new WebSocket(this.config.serverUrl);

        this.ws.onopen = () => {
          this.setState('connected');
          this.reconnectAttempts = 0;
          this.startHeartbeat();

          // Send authentication if token provided
          if (this.config.authToken) {
            this.send({
              type: 'auth',
              authToken: this.config.authToken,
            });
          }

          // Flush pending messages
          this.flushPendingMessages();
          resolve();
        };

        this.ws.onclose = () => {
          this.handleDisconnect();
        };

        this.ws.onerror = (error) => {
          this.emit('error', new Error('WebSocket error'));
          if (this.state === 'connecting') {
            reject(new Error('Failed to connect'));
          }
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };
      } catch (error) {
        this.setState('disconnected');
        reject(error);
      }
    });
  }

  /** Disconnect from the server. */
  disconnect(): void {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setState('disconnected');
  }

  /** Join a document for collaboration. */
  async joinDocument(docId: DocumentId): Promise<CRDTDocument> {
    // Create or get document
    let doc = this.documents.get(docId);
    if (!doc) {
      doc = new CRDTDocument(docId, this.clientId);
      this.documents.set(docId, doc);
    }

    this.currentDocId = docId;

    // Send join message
    this.send({
      type: 'join',
      docId,
    });

    return doc;
  }

  /** Leave the current document. */
  leaveDocument(): void {
    if (this.currentDocId) {
      this.send({
        type: 'leave',
        docId: this.currentDocId,
      });
      this.presence.clear();
      this.currentDocId = null;
    }
  }

  /** Insert text at the current cursor position or after a specific position. */
  insert(char: string, afterPosition?: any): Operation | null {
    const doc = this.currentDocument;
    if (!doc) return null;

    const operation = doc.insert(char, afterPosition);
    this.sendOperation(operation);
    return operation;
  }

  /** Insert text at a specific index. */
  insertAt(index: number, text: string): Operation[] {
    const doc = this.currentDocument;
    if (!doc) return [];

    const operations: Operation[] = [];
    let prevPosition = index > 0 ? doc.getPositionAtIndex(index - 1) : undefined;

    for (const char of text) {
      const operation = doc.insert(char, prevPosition || undefined);
      operations.push(operation);
      prevPosition = operation.position;
    }

    this.sendOperations(operations);
    return operations;
  }

  /** Delete character at position. */
  delete(position: any): Operation | null {
    const doc = this.currentDocument;
    if (!doc) return null;

    const operation = doc.delete(position);
    if (operation) {
      this.sendOperation(operation);
    }
    return operation;
  }

  /** Delete character at index. */
  deleteAt(index: number): Operation | null {
    const doc = this.currentDocument;
    if (!doc) return null;

    const position = doc.getPositionAtIndex(index);
    if (!position) return null;

    return this.delete(position);
  }

  /** Delete a range of characters. */
  deleteRange(startIndex: number, endIndex: number): Operation[] {
    const doc = this.currentDocument;
    if (!doc) return [];

    const operations: Operation[] = [];

    // Delete from end to start to preserve indices
    for (let i = endIndex - 1; i >= startIndex; i--) {
      const position = doc.getPositionAtIndex(i);
      if (position) {
        const operation = doc.delete(position);
        if (operation) {
          operations.push(operation);
        }
      }
    }

    if (operations.length > 0) {
      this.sendOperations(operations);
    }
    return operations;
  }

  /** Format text at position. */
  format(position: any, attributes: Record<string, unknown>): Operation | null {
    const doc = this.currentDocument;
    if (!doc) return null;

    const operation = doc.format(position, attributes);
    if (operation) {
      this.sendOperation(operation);
    }
    return operation;
  }

  /** Update cursor position. */
  updateCursor(cursor: CursorPosition): void {
    this.send({
      type: 'cursor_update',
      docId: this.currentDocId || undefined,
      cursor,
    });
  }

  /** Update selection. */
  updateSelection(selection: Selection): void {
    this.send({
      type: 'cursor_update',
      docId: this.currentDocId || undefined,
      selection,
    });
  }

  /** Update presence status. */
  updatePresence(presence: Partial<PresenceState>): void {
    this.send({
      type: 'presence_update',
      docId: this.currentDocId || undefined,
      presence,
    });
  }

  /** Get document text. */
  getText(): string {
    return this.currentDocument?.getText() || '';
  }

  /** Get document length. */
  getLength(): number {
    return this.currentDocument?.getLength() || 0;
  }

  // Private methods

  private setState(state: ConnectionState): void {
    if (this.state !== state) {
      this.state = state;
      this.emit('connectionStateChange', state);
    }
  }

  private send(message: ClientMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      this.pendingMessages.push(message);
    }
  }

  private sendOperation(operation: Operation): void {
    this.operationSeq++;
    this.pendingAcks.add(this.operationSeq);

    this.send({
      type: 'operation',
      docId: this.currentDocId || undefined,
      operations: [operation],
    });
  }

  private sendOperations(operations: Operation[]): void {
    if (operations.length === 0) return;

    this.operationSeq++;
    this.pendingAcks.add(this.operationSeq);

    this.send({
      type: 'operations_batch',
      docId: this.currentDocId || undefined,
      operations,
    });
  }

  private flushPendingMessages(): void {
    const messages = this.pendingMessages;
    this.pendingMessages = [];
    for (const message of messages) {
      this.send(message);
    }
  }

  private handleMessage(data: string): void {
    try {
      const message: ServerMessage = JSON.parse(data);

      switch (message.type) {
        case 'welcome':
          // Connected successfully
          break;

        case 'doc_state':
          if (message.snapshot) {
            this.handleDocumentState(message.snapshot);
          }
          if (message.presence) {
            this.handlePresenceList(message.presence);
          }
          break;

        case 'operation':
          if (message.operations && message.operations.length > 0) {
            this.handleRemoteOperation(message.operations[0]);
          }
          break;

        case 'operations_batch':
          if (message.operations) {
            this.handleRemoteOperations(message.operations);
          }
          break;

        case 'presence_update':
          if (message.presence) {
            this.handlePresenceList(message.presence);
          }
          break;

        case 'user_joined':
          if (message.presence && message.presence.length > 0) {
            const user = message.presence[0];
            this.presence.set(user.userId, user);
            this.emit('userJoined', user);
          }
          break;

        case 'user_left':
          if (message.clientId) {
            this.presence.delete(message.clientId);
            this.emit('userLeft', message.clientId);
          }
          break;

        case 'ack':
          if (message.seq !== undefined) {
            this.pendingAcks.delete(message.seq);
            this.emit('operationAck', message.seq);
          }
          break;

        case 'error':
          this.emit('error', new Error(message.error || 'Unknown error'));
          break;
      }
    } catch (error) {
      this.emit('error', new Error('Failed to parse server message'));
    }
  }

  private handleDocumentState(snapshot: DocumentSnapshot): void {
    const doc = this.documents.get(snapshot.id);
    if (doc) {
      doc.loadSnapshot(snapshot.content, snapshot.vectorClock);
      this.emit('documentState', snapshot);
    }
  }

  private handleRemoteOperation(operation: Operation): void {
    // Ignore our own operations
    if (operation.clientId === this.clientId) return;

    const doc = this.currentDocument;
    if (doc) {
      doc.applyRemoteOperation(operation);
      this.emit('remoteOperation', operation);
    }
  }

  private handleRemoteOperations(operations: Operation[]): void {
    const doc = this.currentDocument;
    if (!doc) return;

    const remoteOps = operations.filter((op) => op.clientId !== this.clientId);

    for (const operation of remoteOps) {
      doc.applyRemoteOperation(operation);
    }

    if (remoteOps.length > 0) {
      this.emit('remoteOperations', remoteOps);
    }
  }

  private handlePresenceList(presenceList: PresenceState[]): void {
    for (const presence of presenceList) {
      this.presence.set(presence.userId, presence);
    }
    this.emit('presenceUpdate', presenceList);
  }

  private handleDisconnect(): void {
    this.stopHeartbeat();
    this.ws = null;

    if (this.reconnectAttempts < (this.config.maxReconnectAttempts || 5)) {
      this.setState('reconnecting');
      this.reconnectAttempts++;

      const delay = (this.config.reconnectDelay || 1000) * Math.pow(2, this.reconnectAttempts - 1);

      setTimeout(() => {
        this.connect().catch(() => {
          // Reconnection failed, will try again
        });
      }, delay);
    } else {
      this.setState('disconnected');
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.send({ type: 'ping' });
    }, this.config.heartbeatInterval || 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
}

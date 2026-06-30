/**
 * Type definitions for CRDT collaboration client.
 */

/** Client identifier (UUID string). */
export type ClientId = string;

/** Document identifier (UUID string). */
export type DocumentId = string;

/** Position in a CRDT document. */
export interface PositionId {
  lamport: number;
  clientId: ClientId;
  seq: number;
}

/** Vector clock for causality tracking. */
export interface VectorClock {
  clocks: Record<ClientId, number>;
}

/** Operation types. */
export type OperationType = 'insert' | 'delete' | 'format';

/** CRDT operation. */
export interface Operation {
  type: OperationType;
  clientId: ClientId;
  lamport: number;
  position?: PositionId;
  afterPosition?: PositionId;
  char?: string;
  attributes?: Record<string, unknown>;
}

/** Cursor position in a document. */
export interface CursorPosition {
  position: PositionId;
  offset: number;
}

/** Selection range in a document. */
export interface Selection {
  anchor: CursorPosition;
  head: CursorPosition;
}

/** User presence state. */
export interface PresenceState {
  userId: ClientId;
  userName: string;
  cursor?: CursorPosition;
  selection?: Selection;
  color: string;
  status: 'active' | 'idle' | 'away';
  lastActivity: number;
}

/** Document metadata. */
export interface DocumentMetadata {
  id: DocumentId;
  title: string;
  version: number;
  createdAt: number;
  updatedAt: number;
  ownerId: ClientId;
}

/** Document snapshot. */
export interface DocumentSnapshot {
  id: DocumentId;
  content: string;
  vectorClock: VectorClock;
  version: number;
  timestamp: number;
}

/** Permission levels. */
export type Permission = 'read' | 'write' | 'comment' | 'admin';

/** ACL entry. */
export interface AclEntry {
  principal: string;
  permissions: Permission[];
  grantedBy: ClientId;
  grantedAt: number;
  expiresAt?: number;
}

/** Document ACL. */
export interface DocumentAcl {
  docId: DocumentId;
  owner: ClientId;
  entries: AclEntry[];
  publicAccess?: Permission;
}

/** Connection state. */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

/** Client configuration. */
export interface ClientConfig {
  /** Server URL. */
  serverUrl: string;
  /** Client ID (auto-generated if not provided). */
  clientId?: ClientId;
  /** User name for presence. */
  userName?: string;
  /** User color for presence (auto-generated if not provided). */
  userColor?: string;
  /** Authentication token. */
  authToken?: string;
  /** Reconnection attempts (default: 5). */
  maxReconnectAttempts?: number;
  /** Reconnection delay in ms (default: 1000). */
  reconnectDelay?: number;
  /** Heartbeat interval in ms (default: 30000). */
  heartbeatInterval?: number;
}

/** Server message types. */
export type ServerMessageType =
  | 'welcome'
  | 'doc_state'
  | 'operation'
  | 'operations_batch'
  | 'presence_update'
  | 'cursor_update'
  | 'user_joined'
  | 'user_left'
  | 'ack'
  | 'error';

/** Server message. */
export interface ServerMessage {
  type: ServerMessageType;
  docId?: DocumentId;
  clientId?: ClientId;
  operations?: Operation[];
  presence?: PresenceState[];
  snapshot?: DocumentSnapshot;
  seq?: number;
  error?: string;
}

/** Client message types. */
export type ClientMessageType =
  | 'auth'
  | 'join'
  | 'leave'
  | 'operation'
  | 'operations_batch'
  | 'cursor_update'
  | 'presence_update'
  | 'ping';

/** Client message. */
export interface ClientMessage {
  type: ClientMessageType;
  docId?: DocumentId;
  authToken?: string;
  operations?: Operation[];
  cursor?: CursorPosition;
  selection?: Selection;
  presence?: Partial<PresenceState>;
}

/** Client events. */
export interface ClientEvents {
  /** Connection state changed. */
  connectionStateChange: (state: ConnectionState) => void;
  /** Document state received. */
  documentState: (snapshot: DocumentSnapshot) => void;
  /** Remote operation received. */
  remoteOperation: (operation: Operation) => void;
  /** Remote operations batch received. */
  remoteOperations: (operations: Operation[]) => void;
  /** Presence updated. */
  presenceUpdate: (presence: PresenceState[]) => void;
  /** User joined the document. */
  userJoined: (presence: PresenceState) => void;
  /** User left the document. */
  userLeft: (userId: ClientId) => void;
  /** Operation acknowledged by server. */
  operationAck: (seq: number) => void;
  /** Error occurred. */
  error: (error: Error) => void;
}

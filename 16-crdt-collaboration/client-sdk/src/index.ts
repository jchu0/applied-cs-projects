/**
 * CRDT Collaboration Client SDK
 *
 * A TypeScript client for real-time collaborative editing using CRDTs.
 *
 * @example
 * ```typescript
 * import { CollaborationClient } from '@crdt-collaboration/client';
 *
 * const client = new CollaborationClient({
 *   serverUrl: 'ws://localhost:8080',
 *   userName: 'Alice',
 * });
 *
 * await client.connect();
 * const doc = await client.joinDocument('doc-123');
 *
 * // Insert text
 * client.insertAt(0, 'Hello, world!');
 *
 * // Listen for remote changes
 * client.on('remoteOperation', (op) => {
 *   console.log('Remote change:', op);
 * });
 *
 * // Listen for presence updates
 * client.on('presenceUpdate', (users) => {
 *   console.log('Users:', users);
 * });
 * ```
 */

export { CollaborationClient } from './client';
export { CRDTDocument, generateUUID, comparePositionIds, positionIdsEqual } from './crdt';
export type { CRDTElement } from './crdt';

export type {
  // Core types
  ClientId,
  DocumentId,
  PositionId,
  VectorClock,
  Operation,
  OperationType,

  // Presence types
  CursorPosition,
  Selection,
  PresenceState,

  // Document types
  DocumentMetadata,
  DocumentSnapshot,

  // Permission types
  Permission,
  AclEntry,
  DocumentAcl,

  // Connection types
  ConnectionState,
  ClientConfig,
  ClientEvents,

  // Message types
  ServerMessage,
  ServerMessageType,
  ClientMessage,
  ClientMessageType,
} from './types';

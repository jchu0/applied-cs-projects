/**
 * CRDT implementation for client-side document representation.
 */

import {
  ClientId,
  DocumentId,
  PositionId,
  VectorClock,
  Operation,
  OperationType,
} from './types';

/** Generate a UUID v4. */
export function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** CRDT element representing a character or object in the document. */
export interface CRDTElement {
  id: PositionId;
  char: string;
  attributes: Record<string, unknown>;
  deleted: boolean;
  leftId?: PositionId;
  rightId?: PositionId;
}

/** Compare two position IDs. Returns -1, 0, or 1. */
export function comparePositionIds(a: PositionId, b: PositionId): number {
  if (a.lamport !== b.lamport) {
    return a.lamport < b.lamport ? -1 : 1;
  }
  if (a.clientId !== b.clientId) {
    return a.clientId < b.clientId ? -1 : 1;
  }
  return a.seq < b.seq ? -1 : a.seq > b.seq ? 1 : 0;
}

/** Check if two position IDs are equal. */
export function positionIdsEqual(a: PositionId, b: PositionId): boolean {
  return a.lamport === b.lamport && a.clientId === b.clientId && a.seq === b.seq;
}

/** Client-side CRDT document. */
export class CRDTDocument {
  private docId: DocumentId;
  private clientId: ClientId;
  private lamport: number = 0;
  private seq: number = 0;
  private elements: Map<string, CRDTElement> = new Map();
  private head: PositionId | null = null;
  private vectorClock: VectorClock = { clocks: {} };
  private pendingOperations: Operation[] = [];

  constructor(docId: DocumentId, clientId: ClientId) {
    this.docId = docId;
    this.clientId = clientId;
  }

  /** Get current vector clock. */
  getVectorClock(): VectorClock {
    return { ...this.vectorClock, clocks: { ...this.vectorClock.clocks } };
  }

  /** Update vector clock from another clock. */
  updateVectorClock(other: VectorClock): void {
    for (const [clientId, lamport] of Object.entries(other.clocks)) {
      const current = this.vectorClock.clocks[clientId] || 0;
      this.vectorClock.clocks[clientId] = Math.max(current, lamport);
    }
  }

  /** Get position ID key for map lookup. */
  private positionKey(pos: PositionId): string {
    return `${pos.lamport}:${pos.clientId}:${pos.seq}`;
  }

  /** Increment and get next lamport timestamp. */
  private nextLamport(): number {
    this.lamport += 1;
    return this.lamport;
  }

  /** Insert a character after a position. */
  insert(char: string, afterPosition?: PositionId): Operation {
    const lamport = this.nextLamport();
    this.seq += 1;

    const position: PositionId = {
      lamport,
      clientId: this.clientId,
      seq: this.seq,
    };

    const element: CRDTElement = {
      id: position,
      char,
      attributes: {},
      deleted: false,
      leftId: afterPosition,
    };

    // Find the right neighbor
    if (afterPosition) {
      const afterKey = this.positionKey(afterPosition);
      const afterElement = this.elements.get(afterKey);
      if (afterElement) {
        element.rightId = afterElement.rightId;
        afterElement.rightId = position;
      }
    } else if (this.head) {
      // Insert at beginning
      element.rightId = this.head;
      this.head = position;
    } else {
      // First element
      this.head = position;
    }

    this.elements.set(this.positionKey(position), element);

    // Update vector clock
    this.vectorClock.clocks[this.clientId] = lamport;

    const operation: Operation = {
      type: 'insert',
      clientId: this.clientId,
      lamport,
      position,
      afterPosition,
      char,
    };

    this.pendingOperations.push(operation);
    return operation;
  }

  /** Delete a character at a position. */
  delete(position: PositionId): Operation | null {
    const key = this.positionKey(position);
    const element = this.elements.get(key);

    if (!element || element.deleted) {
      return null;
    }

    element.deleted = true;
    const lamport = this.nextLamport();

    // Update vector clock
    this.vectorClock.clocks[this.clientId] = lamport;

    const operation: Operation = {
      type: 'delete',
      clientId: this.clientId,
      lamport,
      position,
    };

    this.pendingOperations.push(operation);
    return operation;
  }

  /** Format text at a position. */
  format(position: PositionId, attributes: Record<string, unknown>): Operation | null {
    const key = this.positionKey(position);
    const element = this.elements.get(key);

    if (!element || element.deleted) {
      return null;
    }

    // Merge attributes
    element.attributes = { ...element.attributes, ...attributes };
    const lamport = this.nextLamport();

    // Update vector clock
    this.vectorClock.clocks[this.clientId] = lamport;

    const operation: Operation = {
      type: 'format',
      clientId: this.clientId,
      lamport,
      position,
      attributes,
    };

    this.pendingOperations.push(operation);
    return operation;
  }

  /** Apply a remote operation. */
  applyRemoteOperation(operation: Operation): void {
    // Update lamport clock
    this.lamport = Math.max(this.lamport, operation.lamport);

    // Update vector clock
    const currentClock = this.vectorClock.clocks[operation.clientId] || 0;
    this.vectorClock.clocks[operation.clientId] = Math.max(currentClock, operation.lamport);

    switch (operation.type) {
      case 'insert':
        this.applyRemoteInsert(operation);
        break;
      case 'delete':
        this.applyRemoteDelete(operation);
        break;
      case 'format':
        this.applyRemoteFormat(operation);
        break;
    }
  }

  private applyRemoteInsert(operation: Operation): void {
    if (!operation.position || !operation.char) return;

    const key = this.positionKey(operation.position);
    if (this.elements.has(key)) {
      // Already have this operation (idempotent)
      return;
    }

    const element: CRDTElement = {
      id: operation.position,
      char: operation.char,
      attributes: operation.attributes || {},
      deleted: false,
      leftId: operation.afterPosition,
    };

    // Find correct position based on CRDT ordering
    if (operation.afterPosition) {
      const afterKey = this.positionKey(operation.afterPosition);
      const afterElement = this.elements.get(afterKey);

      if (afterElement) {
        // Find correct position among concurrent inserts
        let currentRight = afterElement.rightId;
        while (currentRight) {
          const rightKey = this.positionKey(currentRight);
          const rightElement = this.elements.get(rightKey);
          if (!rightElement) break;

          // Check if we should insert before this element
          if (comparePositionIds(operation.position, currentRight) < 0) {
            break;
          }
          currentRight = rightElement.rightId;
        }

        element.rightId = currentRight;
        afterElement.rightId = operation.position;
      }
    } else {
      // Insert at beginning
      if (!this.head || comparePositionIds(operation.position, this.head) < 0) {
        element.rightId = this.head;
        this.head = operation.position;
      }
    }

    this.elements.set(key, element);
  }

  private applyRemoteDelete(operation: Operation): void {
    if (!operation.position) return;

    const key = this.positionKey(operation.position);
    const element = this.elements.get(key);

    if (element) {
      element.deleted = true;
    }
  }

  private applyRemoteFormat(operation: Operation): void {
    if (!operation.position || !operation.attributes) return;

    const key = this.positionKey(operation.position);
    const element = this.elements.get(key);

    if (element && !element.deleted) {
      element.attributes = { ...element.attributes, ...operation.attributes };
    }
  }

  /** Get pending operations and clear the buffer. */
  flushPendingOperations(): Operation[] {
    const operations = this.pendingOperations;
    this.pendingOperations = [];
    return operations;
  }

  /** Get document content as plain text. */
  getText(): string {
    const chars: string[] = [];
    let current = this.head;

    while (current) {
      const key = this.positionKey(current);
      const element = this.elements.get(key);
      if (!element) break;

      if (!element.deleted) {
        chars.push(element.char);
      }

      current = element.rightId || null;
    }

    return chars.join('');
  }

  /** Get document content with position information. */
  getContent(): Array<{ char: string; position: PositionId; attributes: Record<string, unknown> }> {
    const content: Array<{ char: string; position: PositionId; attributes: Record<string, unknown> }> = [];
    let current = this.head;

    while (current) {
      const key = this.positionKey(current);
      const element = this.elements.get(key);
      if (!element) break;

      if (!element.deleted) {
        content.push({
          char: element.char,
          position: element.id,
          attributes: element.attributes,
        });
      }

      current = element.rightId || null;
    }

    return content;
  }

  /** Get position at a given index. */
  getPositionAtIndex(index: number): PositionId | null {
    let current = this.head;
    let i = 0;

    while (current) {
      const key = this.positionKey(current);
      const element = this.elements.get(key);
      if (!element) break;

      if (!element.deleted) {
        if (i === index) {
          return element.id;
        }
        i++;
      }

      current = element.rightId || null;
    }

    return null;
  }

  /** Get index of a position. */
  getIndexOfPosition(position: PositionId): number {
    let current = this.head;
    let index = 0;

    while (current) {
      const key = this.positionKey(current);
      const element = this.elements.get(key);
      if (!element) break;

      if (positionIdsEqual(element.id, position)) {
        return index;
      }

      if (!element.deleted) {
        index++;
      }

      current = element.rightId || null;
    }

    return -1;
  }

  /** Get document length (non-deleted characters). */
  getLength(): number {
    let length = 0;
    let current = this.head;

    while (current) {
      const key = this.positionKey(current);
      const element = this.elements.get(key);
      if (!element) break;

      if (!element.deleted) {
        length++;
      }

      current = element.rightId || null;
    }

    return length;
  }

  /** Load document from snapshot. */
  loadSnapshot(content: string, vectorClock: VectorClock): void {
    this.elements.clear();
    this.head = null;
    this.updateVectorClock(vectorClock);

    let prevPosition: PositionId | undefined;

    for (let i = 0; i < content.length; i++) {
      const char = content[i];
      const position: PositionId = {
        lamport: i + 1,
        clientId: 'snapshot',
        seq: i + 1,
      };

      const element: CRDTElement = {
        id: position,
        char,
        attributes: {},
        deleted: false,
        leftId: prevPosition,
      };

      if (prevPosition) {
        const prevKey = this.positionKey(prevPosition);
        const prevElement = this.elements.get(prevKey);
        if (prevElement) {
          prevElement.rightId = position;
        }
      } else {
        this.head = position;
      }

      this.elements.set(this.positionKey(position), element);
      prevPosition = position;
    }

    this.lamport = Math.max(this.lamport, content.length);
  }
}

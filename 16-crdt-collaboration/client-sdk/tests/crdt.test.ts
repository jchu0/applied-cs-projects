/**
 * Tests for CRDT implementation.
 */

import {
  CRDTDocument,
  comparePositionIds,
  positionIdsEqual,
  generateUUID,
} from '../src/crdt';
import { PositionId } from '../src/types';

describe('Position ID utilities', () => {
  test('comparePositionIds compares by lamport first', () => {
    const a: PositionId = { lamport: 1, clientId: 'a', seq: 1 };
    const b: PositionId = { lamport: 2, clientId: 'a', seq: 1 };

    expect(comparePositionIds(a, b)).toBe(-1);
    expect(comparePositionIds(b, a)).toBe(1);
  });

  test('comparePositionIds compares by clientId when lamport equal', () => {
    const a: PositionId = { lamport: 1, clientId: 'a', seq: 1 };
    const b: PositionId = { lamport: 1, clientId: 'b', seq: 1 };

    expect(comparePositionIds(a, b)).toBe(-1);
    expect(comparePositionIds(b, a)).toBe(1);
  });

  test('comparePositionIds compares by seq when lamport and clientId equal', () => {
    const a: PositionId = { lamport: 1, clientId: 'a', seq: 1 };
    const b: PositionId = { lamport: 1, clientId: 'a', seq: 2 };

    expect(comparePositionIds(a, b)).toBe(-1);
    expect(comparePositionIds(b, a)).toBe(1);
  });

  test('positionIdsEqual returns true for equal positions', () => {
    const a: PositionId = { lamport: 1, clientId: 'a', seq: 1 };
    const b: PositionId = { lamport: 1, clientId: 'a', seq: 1 };

    expect(positionIdsEqual(a, b)).toBe(true);
  });

  test('positionIdsEqual returns false for different positions', () => {
    const a: PositionId = { lamport: 1, clientId: 'a', seq: 1 };
    const b: PositionId = { lamport: 2, clientId: 'a', seq: 1 };

    expect(positionIdsEqual(a, b)).toBe(false);
  });

  test('generateUUID generates valid UUID', () => {
    const uuid = generateUUID();
    expect(uuid).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
  });
});

describe('CRDTDocument', () => {
  let doc: CRDTDocument;

  beforeEach(() => {
    doc = new CRDTDocument('doc-1', 'client-1');
  });

  test('starts empty', () => {
    expect(doc.getText()).toBe('');
    expect(doc.getLength()).toBe(0);
  });

  test('insert single character', () => {
    const op = doc.insert('a');

    expect(doc.getText()).toBe('a');
    expect(doc.getLength()).toBe(1);
    expect(op.type).toBe('insert');
    expect(op.char).toBe('a');
  });

  test('insert multiple characters', () => {
    doc.insert('H');
    const op2 = doc.insert('e', doc.getPositionAtIndex(0)!);
    doc.insert('l', op2.position!);
    doc.insert('l', doc.getPositionAtIndex(2)!);
    doc.insert('o', doc.getPositionAtIndex(3)!);

    expect(doc.getText()).toBe('Hello');
    expect(doc.getLength()).toBe(5);
  });

  test('delete character', () => {
    doc.insert('a');
    doc.insert('b', doc.getPositionAtIndex(0)!);
    doc.insert('c', doc.getPositionAtIndex(1)!);

    expect(doc.getText()).toBe('abc');

    const pos = doc.getPositionAtIndex(1)!;
    doc.delete(pos);

    expect(doc.getText()).toBe('ac');
    expect(doc.getLength()).toBe(2);
  });

  test('delete returns null for already deleted', () => {
    const op = doc.insert('a');
    const pos = op.position!;

    doc.delete(pos);
    const result = doc.delete(pos);

    expect(result).toBeNull();
  });

  test('format character', () => {
    const op = doc.insert('a');
    const formatOp = doc.format(op.position!, { bold: true });

    expect(formatOp).not.toBeNull();
    expect(formatOp!.type).toBe('format');
    expect(formatOp!.attributes).toEqual({ bold: true });
  });

  test('getPositionAtIndex returns null for out of bounds', () => {
    doc.insert('a');

    expect(doc.getPositionAtIndex(5)).toBeNull();
  });

  test('getIndexOfPosition returns -1 for missing position', () => {
    doc.insert('a');

    const fakePos: PositionId = { lamport: 999, clientId: 'fake', seq: 1 };
    expect(doc.getIndexOfPosition(fakePos)).toBe(-1);
  });

  test('flushPendingOperations returns and clears operations', () => {
    doc.insert('a');
    doc.insert('b', doc.getPositionAtIndex(0)!);

    const pending = doc.flushPendingOperations();
    expect(pending).toHaveLength(2);

    const pending2 = doc.flushPendingOperations();
    expect(pending2).toHaveLength(0);
  });

  test('vector clock updates on local operations', () => {
    const initialClock = doc.getVectorClock();
    expect(initialClock.clocks).toEqual({});

    doc.insert('a');
    const clock1 = doc.getVectorClock();
    expect(clock1.clocks['client-1']).toBe(1);

    doc.insert('b', doc.getPositionAtIndex(0)!);
    const clock2 = doc.getVectorClock();
    expect(clock2.clocks['client-1']).toBe(2);
  });

  test('getContent returns characters with positions', () => {
    const op1 = doc.insert('a');
    const op2 = doc.insert('b', op1.position!);

    const content = doc.getContent();
    expect(content).toHaveLength(2);
    expect(content[0].char).toBe('a');
    expect(content[1].char).toBe('b');
    expect(content[0].position).toBeDefined();
    expect(content[1].position).toBeDefined();
  });
});

describe('CRDTDocument - Remote operations', () => {
  test('apply remote insert', () => {
    const doc1 = new CRDTDocument('doc-1', 'client-1');
    const doc2 = new CRDTDocument('doc-1', 'client-2');

    const op = doc1.insert('a');

    doc2.applyRemoteOperation(op);

    expect(doc2.getText()).toBe('a');
  });

  test('apply remote delete', () => {
    const doc1 = new CRDTDocument('doc-1', 'client-1');
    const doc2 = new CRDTDocument('doc-1', 'client-2');

    const insertOp = doc1.insert('a');
    doc2.applyRemoteOperation(insertOp);

    const deleteOp = doc1.delete(insertOp.position!);
    doc2.applyRemoteOperation(deleteOp!);

    expect(doc2.getText()).toBe('');
  });

  test('concurrent inserts converge', () => {
    const doc1 = new CRDTDocument('doc-1', 'client-1');
    const doc2 = new CRDTDocument('doc-1', 'client-2');

    // Client 1 inserts 'a'
    const op1 = doc1.insert('a');

    // Client 2 inserts 'b' (concurrent)
    const op2 = doc2.insert('b');

    // Apply ops to both docs
    doc1.applyRemoteOperation(op2);
    doc2.applyRemoteOperation(op1);

    // Both docs should have same content (order may vary by CRDT rules)
    expect(doc1.getText()).toBe(doc2.getText());
    expect(doc1.getLength()).toBe(2);
    expect(doc2.getLength()).toBe(2);
  });

  test('idempotent remote inserts', () => {
    const doc1 = new CRDTDocument('doc-1', 'client-1');
    const doc2 = new CRDTDocument('doc-1', 'client-2');

    const op = doc1.insert('a');

    // Apply same operation twice
    doc2.applyRemoteOperation(op);
    doc2.applyRemoteOperation(op);

    expect(doc2.getText()).toBe('a');
    expect(doc2.getLength()).toBe(1);
  });

  test('apply remote format', () => {
    const doc1 = new CRDTDocument('doc-1', 'client-1');
    const doc2 = new CRDTDocument('doc-1', 'client-2');

    const insertOp = doc1.insert('a');
    doc2.applyRemoteOperation(insertOp);

    const formatOp = doc1.format(insertOp.position!, { bold: true });
    doc2.applyRemoteOperation(formatOp!);

    const content = doc2.getContent();
    expect(content[0].attributes).toEqual({ bold: true });
  });

  test('vector clock updates on remote operations', () => {
    const doc = new CRDTDocument('doc-1', 'client-1');

    doc.applyRemoteOperation({
      type: 'insert',
      clientId: 'client-2',
      lamport: 5,
      position: { lamport: 5, clientId: 'client-2', seq: 1 },
      char: 'x',
    });

    const clock = doc.getVectorClock();
    expect(clock.clocks['client-2']).toBe(5);
  });
});

describe('CRDTDocument - Load snapshot', () => {
  test('loadSnapshot initializes document', () => {
    const doc = new CRDTDocument('doc-1', 'client-1');

    doc.loadSnapshot('Hello', { clocks: { snapshot: 5 } });

    expect(doc.getText()).toBe('Hello');
    expect(doc.getLength()).toBe(5);
  });

  test('loadSnapshot clears existing content', () => {
    const doc = new CRDTDocument('doc-1', 'client-1');

    doc.insert('a');
    doc.insert('b', doc.getPositionAtIndex(0)!);

    doc.loadSnapshot('xyz', { clocks: {} });

    expect(doc.getText()).toBe('xyz');
    expect(doc.getLength()).toBe(3);
  });

  test('operations after snapshot work correctly', () => {
    const doc = new CRDTDocument('doc-1', 'client-1');

    doc.loadSnapshot('Hello', { clocks: {} });

    const pos = doc.getPositionAtIndex(4)!; // After 'o'
    doc.insert('!', pos);

    expect(doc.getText()).toBe('Hello!');
  });
});

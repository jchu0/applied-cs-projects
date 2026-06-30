/**
 * Tests for CollaborationClient.
 */

import { CollaborationClient } from '../src/client';

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: ((error: any) => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;

  sentMessages: string[] = [];

  constructor(public url: string) {
    // Simulate async connection
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) this.onopen();
    }, 0);
  }

  send(data: string): void {
    this.sentMessages.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) this.onclose();
  }

  // Test helper to simulate receiving a message
  receiveMessage(data: any): void {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }
}

// @ts-ignore - Mock global WebSocket
global.WebSocket = MockWebSocket as any;

describe('CollaborationClient', () => {
  let client: CollaborationClient;

  beforeEach(() => {
    client = new CollaborationClient({
      serverUrl: 'ws://localhost:8080',
      userName: 'TestUser',
    });
  });

  afterEach(() => {
    client.disconnect();
  });

  describe('initialization', () => {
    test('generates client ID if not provided', () => {
      expect(client.clientId).toBeDefined();
      expect(client.clientId).toMatch(/^[0-9a-f-]{36}$/i);
    });

    test('uses provided client ID', () => {
      const customClient = new CollaborationClient({
        serverUrl: 'ws://localhost:8080',
        clientId: 'custom-id',
      });
      expect(customClient.clientId).toBe('custom-id');
    });

    test('starts disconnected', () => {
      expect(client.connectionState).toBe('disconnected');
    });
  });

  describe('connection', () => {
    test('connect changes state to connecting then connected', async () => {
      const states: string[] = [];
      client.on('connectionStateChange', (state) => {
        states.push(state);
      });

      await client.connect();

      expect(states).toContain('connecting');
      expect(states).toContain('connected');
      expect(client.connectionState).toBe('connected');
    });

    test('disconnect changes state to disconnected', async () => {
      await client.connect();
      client.disconnect();

      expect(client.connectionState).toBe('disconnected');
    });

    test('sends auth message if token provided', async () => {
      const authClient = new CollaborationClient({
        serverUrl: 'ws://localhost:8080',
        authToken: 'test-token',
      });

      await authClient.connect();

      // Get the underlying mock WebSocket
      const ws = (authClient as any).ws as MockWebSocket;
      const authMessage = ws.sentMessages.find((m) =>
        JSON.parse(m).type === 'auth'
      );

      expect(authMessage).toBeDefined();
      expect(JSON.parse(authMessage!).authToken).toBe('test-token');
    });
  });

  describe('document operations', () => {
    beforeEach(async () => {
      await client.connect();
    });

    test('joinDocument creates document and sends join message', async () => {
      const doc = await client.joinDocument('doc-123');

      expect(doc).toBeDefined();
      expect(client.currentDocument).toBe(doc);

      const ws = (client as any).ws as MockWebSocket;
      const joinMessage = ws.sentMessages.find((m) =>
        JSON.parse(m).type === 'join'
      );

      expect(joinMessage).toBeDefined();
      expect(JSON.parse(joinMessage!).docId).toBe('doc-123');
    });

    test('leaveDocument clears current document', async () => {
      await client.joinDocument('doc-123');
      client.leaveDocument();

      expect(client.currentDocument).toBeNull();
    });

    test('insert returns operation', async () => {
      await client.joinDocument('doc-123');
      const op = client.insert('a');

      expect(op).toBeDefined();
      expect(op!.type).toBe('insert');
      expect(op!.char).toBe('a');
    });

    test('insertAt inserts text at index', async () => {
      await client.joinDocument('doc-123');

      const ops = client.insertAt(0, 'Hello');

      expect(ops).toHaveLength(5);
      expect(client.getText()).toBe('Hello');
    });

    test('deleteAt removes character', async () => {
      await client.joinDocument('doc-123');
      client.insertAt(0, 'abc');

      client.deleteAt(1);

      expect(client.getText()).toBe('ac');
    });

    test('deleteRange removes multiple characters', async () => {
      await client.joinDocument('doc-123');
      client.insertAt(0, 'Hello');

      client.deleteRange(1, 4);

      expect(client.getText()).toBe('Ho');
    });

    test('getText returns document content', async () => {
      await client.joinDocument('doc-123');
      client.insertAt(0, 'Test');

      expect(client.getText()).toBe('Test');
    });

    test('getLength returns document length', async () => {
      await client.joinDocument('doc-123');
      client.insertAt(0, 'Test');

      expect(client.getLength()).toBe(4);
    });
  });

  describe('presence', () => {
    beforeEach(async () => {
      await client.connect();
      await client.joinDocument('doc-123');
    });

    test('updateCursor sends cursor update', () => {
      const cursor = {
        position: { lamport: 1, clientId: client.clientId, seq: 1 },
        offset: 0,
      };

      client.updateCursor(cursor);

      const ws = (client as any).ws as MockWebSocket;
      const cursorMessage = ws.sentMessages.find((m) =>
        JSON.parse(m).type === 'cursor_update'
      );

      expect(cursorMessage).toBeDefined();
    });

    test('updatePresence sends presence update', () => {
      client.updatePresence({ status: 'idle' });

      const ws = (client as any).ws as MockWebSocket;
      const presenceMessage = ws.sentMessages.find((m) =>
        JSON.parse(m).type === 'presence_update'
      );

      expect(presenceMessage).toBeDefined();
    });

    test('users returns empty array initially', () => {
      expect(client.users).toEqual([]);
    });
  });

  describe('server messages', () => {
    beforeEach(async () => {
      await client.connect();
      await client.joinDocument('doc-123');
    });

    test('handles doc_state message', (done) => {
      client.on('documentState', (snapshot) => {
        expect(snapshot.id).toBe('doc-123');
        expect(snapshot.content).toBe('Initial');
        done();
      });

      const ws = (client as any).ws as MockWebSocket;
      ws.receiveMessage({
        type: 'doc_state',
        snapshot: {
          id: 'doc-123',
          content: 'Initial',
          vectorClock: { clocks: {} },
          version: 1,
          timestamp: Date.now(),
        },
      });
    });

    test('handles remote operation', (done) => {
      client.on('remoteOperation', (op) => {
        expect(op.type).toBe('insert');
        expect(op.char).toBe('x');
        done();
      });

      const ws = (client as any).ws as MockWebSocket;
      ws.receiveMessage({
        type: 'operation',
        operations: [
          {
            type: 'insert',
            clientId: 'other-client',
            lamport: 1,
            position: { lamport: 1, clientId: 'other-client', seq: 1 },
            char: 'x',
          },
        ],
      });
    });

    test('ignores own operations in remote handler', () => {
      const handler = jest.fn();
      client.on('remoteOperation', handler);

      const ws = (client as any).ws as MockWebSocket;
      ws.receiveMessage({
        type: 'operation',
        operations: [
          {
            type: 'insert',
            clientId: client.clientId, // Same client
            lamport: 1,
            position: { lamport: 1, clientId: client.clientId, seq: 1 },
            char: 'x',
          },
        ],
      });

      expect(handler).not.toHaveBeenCalled();
    });

    test('handles user_joined message', (done) => {
      client.on('userJoined', (presence) => {
        expect(presence.userId).toBe('user-2');
        expect(presence.userName).toBe('Alice');
        done();
      });

      const ws = (client as any).ws as MockWebSocket;
      ws.receiveMessage({
        type: 'user_joined',
        presence: [
          {
            userId: 'user-2',
            userName: 'Alice',
            color: '#FF0000',
            status: 'active',
            lastActivity: Date.now(),
          },
        ],
      });
    });

    test('handles user_left message', (done) => {
      // First add a user
      const ws = (client as any).ws as MockWebSocket;
      ws.receiveMessage({
        type: 'user_joined',
        presence: [
          {
            userId: 'user-2',
            userName: 'Alice',
            color: '#FF0000',
            status: 'active',
            lastActivity: Date.now(),
          },
        ],
      });

      client.on('userLeft', (userId) => {
        expect(userId).toBe('user-2');
        done();
      });

      ws.receiveMessage({
        type: 'user_left',
        clientId: 'user-2',
      });
    });

    test('handles error message', (done) => {
      client.on('error', (error) => {
        expect(error.message).toBe('Test error');
        done();
      });

      const ws = (client as any).ws as MockWebSocket;
      ws.receiveMessage({
        type: 'error',
        error: 'Test error',
      });
    });

    test('handles ack message', (done) => {
      client.on('operationAck', (seq) => {
        expect(seq).toBe(1);
        done();
      });

      const ws = (client as any).ws as MockWebSocket;
      ws.receiveMessage({
        type: 'ack',
        seq: 1,
      });
    });
  });
});

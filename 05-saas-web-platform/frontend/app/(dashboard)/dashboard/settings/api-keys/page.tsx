'use client';

import { useState } from 'react';

interface APIKey {
  id: string;
  name: string;
  prefix: string;
  lastUsed: string | null;
  createdAt: string;
}

const initialKeys: APIKey[] = [
  { id: '1', name: 'Production API Key', prefix: 'sk_live_', lastUsed: '2 hours ago', createdAt: 'Jan 1, 2024' },
  { id: '2', name: 'Development Key', prefix: 'sk_test_', lastUsed: '1 day ago', createdAt: 'Feb 1, 2024' },
];

export default function APIKeysPage() {
  const [keys, setKeys] = useState<APIKey[]>(initialKeys);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newKeyVisible, setNewKeyVisible] = useState<string | null>(null);

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this API key?')) {
      setKeys(keys.filter(k => k.id !== id));
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">API Keys</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage API keys for programmatic access to your workspace.
          </p>
        </div>
        <button
          onClick={() => setIsCreateModalOpen(true)}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          Create API key
        </button>
      </div>

      {/* New Key Alert */}
      {newKeyVisible && (
        <div className="mt-6 rounded-md bg-green-50 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3 flex-1">
              <h3 className="text-sm font-medium text-green-800">API key created</h3>
              <div className="mt-2">
                <p className="text-sm text-green-700">
                  Make sure to copy your API key now. You won&apos;t be able to see it again!
                </p>
                <div className="mt-2 flex items-center space-x-2">
                  <code className="rounded bg-green-100 px-2 py-1 text-sm text-green-900">
                    {newKeyVisible}
                  </code>
                  <button
                    onClick={() => navigator.clipboard.writeText(newKeyVisible)}
                    className="text-sm font-medium text-green-600 hover:text-green-500"
                  >
                    Copy
                  </button>
                </div>
              </div>
              <div className="mt-4">
                <button
                  onClick={() => setNewKeyVisible(null)}
                  className="text-sm font-medium text-green-600 hover:text-green-500"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* API Keys Table */}
      <div className="mt-8 overflow-hidden rounded-lg bg-white shadow">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Name
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Key
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Last used
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Created
              </th>
              <th className="relative px-6 py-3">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {keys.map((key) => (
              <tr key={key.id}>
                <td className="whitespace-nowrap px-6 py-4 text-sm font-medium text-gray-900">
                  {key.name}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  <code className="rounded bg-gray-100 px-2 py-1">
                    {key.prefix}...
                  </code>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {key.lastUsed || 'Never'}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {key.createdAt}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-right text-sm font-medium">
                  <button
                    onClick={() => handleDelete(key.id)}
                    className="text-red-600 hover:text-red-900"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Create Key Modal */}
      {isCreateModalOpen && (
        <CreateKeyModal
          onClose={() => setIsCreateModalOpen(false)}
          onCreate={(name, key) => {
            setKeys([...keys, {
              id: String(Date.now()),
              name,
              prefix: key.slice(0, 8),
              lastUsed: null,
              createdAt: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
            }]);
            setNewKeyVisible(key);
            setIsCreateModalOpen(false);
          }}
        />
      )}
    </div>
  );
}

function CreateKeyModal({ onClose, onCreate }: { onClose: () => void; onCreate: (name: string, key: string) => void }) {
  const [name, setName] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Generate a mock API key
    const key = 'sk_live_' + Array.from({ length: 32 }, () =>
      'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'[Math.floor(Math.random() * 62)]
    ).join('');
    onCreate(name, key);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
        <h2 className="text-lg font-medium text-gray-900">Create API key</h2>
        <form onSubmit={handleSubmit} className="mt-4">
          <div>
            <label htmlFor="keyName" className="block text-sm font-medium text-gray-700">
              Key name
            </label>
            <input
              type="text"
              id="keyName"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="e.g., Production API Key"
            />
          </div>
          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
            >
              Create key
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

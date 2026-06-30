'use client';

import { useEffect, useState } from 'react';
import { useApi } from '@/lib/hooks/use-api';

interface AuditLog {
  id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  user: string | null;
  tenant: string | null;
  ip_address: string | null;
  created_at: string;
}

interface AuditLogsResponse {
  logs: AuditLog[];
  total: number;
  page: number;
  per_page: number;
}

const actionColors: Record<string, string> = {
  CREATE: 'bg-green-100 text-green-800',
  UPDATE: 'bg-blue-100 text-blue-800',
  DELETE: 'bg-red-100 text-red-800',
  LOGIN: 'bg-purple-100 text-purple-800',
  LOGOUT: 'bg-gray-100 text-gray-800',
  VIEW: 'bg-yellow-100 text-yellow-800',
};

export default function AdminAuditLogsPage() {
  const { get, isLoading, error } = useApi();
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [filterAction, setFilterAction] = useState<string>('');
  const [filterResourceType, setFilterResourceType] = useState<string>('');

  useEffect(() => {
    loadLogs();
  }, [page, filterAction, filterResourceType]);

  const loadLogs = async () => {
    let endpoint = `admin/audit-logs?page=${page}&per_page=50`;
    if (filterAction) {
      endpoint += `&action=${filterAction}`;
    }
    if (filterResourceType) {
      endpoint += `&resource_type=${filterResourceType}`;
    }

    const data = await get(endpoint) as AuditLogsResponse | null;
    if (data) {
      setLogs(data.logs);
      setTotal(data.total);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getActionBadge = (action: string) => {
    return actionColors[action] || 'bg-gray-100 text-gray-800';
  };

  const totalPages = Math.ceil(total / 50);

  return (
    <div>
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Audit Logs</h1>
          <p className="mt-1 text-sm text-gray-500">
            System activity and security events ({total} total)
          </p>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700">
          {error.message}
        </div>
      )}

      {/* Filters */}
      <div className="mt-6 bg-white rounded-lg shadow p-4">
        <div className="flex gap-4 items-end">
          <div>
            <label className="block text-sm font-medium text-gray-700">Action</label>
            <select
              value={filterAction}
              onChange={(e) => {
                setFilterAction(e.target.value);
                setPage(1);
              }}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="">All Actions</option>
              <option value="CREATE">Create</option>
              <option value="UPDATE">Update</option>
              <option value="DELETE">Delete</option>
              <option value="LOGIN">Login</option>
              <option value="LOGOUT">Logout</option>
              <option value="VIEW">View</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Resource Type</label>
            <select
              value={filterResourceType}
              onChange={(e) => {
                setFilterResourceType(e.target.value);
                setPage(1);
              }}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="">All Resources</option>
              <option value="user">User</option>
              <option value="tenant">Tenant</option>
              <option value="subscription">Subscription</option>
              <option value="api_key">API Key</option>
              <option value="invoice">Invoice</option>
            </select>
          </div>
          <button
            onClick={() => {
              setFilterAction('');
              setFilterResourceType('');
              setPage(1);
            }}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
          >
            Clear Filters
          </button>
        </div>
      </div>

      {/* Logs Table */}
      <div className="mt-6 bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Timestamp
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Action
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Resource
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                User
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Tenant
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                IP Address
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {isLoading ? (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-gray-500">
                  Loading...
                </td>
              </tr>
            ) : logs.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-gray-500">
                  No audit logs found
                </td>
              </tr>
            ) : (
              logs.map((log) => (
                <tr key={log.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatDate(log.created_at)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getActionBadge(
                        log.action
                      )}`}
                    >
                      {log.action}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{log.resource_type}</div>
                    <div className="text-xs text-gray-500 font-mono">
                      {log.resource_id ? log.resource_id.substring(0, 8) + '...' : '-'}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {log.user || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {log.tenant || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                    {log.ip_address || '-'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-4 flex justify-between items-center">
          <p className="text-sm text-gray-500">
            Showing {(page - 1) * 50 + 1} to {Math.min(page * 50, total)} of {total} logs
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 border rounded-md disabled:opacity-50"
            >
              Previous
            </button>
            <span className="px-3 py-1">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-3 py-1 border rounded-md disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

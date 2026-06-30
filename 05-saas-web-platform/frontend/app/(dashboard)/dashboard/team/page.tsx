'use client';

import { useState, useEffect } from 'react';
import { useApi } from '@/lib/hooks/use-api';
import { useTenant, usePermission } from '@/lib/hooks/use-tenant';

interface TeamMember {
  id: string;
  user: {
    id: string;
    email: string;
    first_name: string;
    last_name: string;
    full_name?: string;
  } | null;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  invited_email?: string;
  invited_at: string | null;
  accepted_at: string | null;
}

interface Invitation {
  id: string;
  email: string;
  role: string;
  created_at: string;
  expires_at: string;
}

export default function TeamPage() {
  const { get, post, del, isLoading, error } = useApi();
  const { currentTenant } = useTenant();
  const { canInviteMembers, canManageTeam } = usePermission();
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [isInviteModalOpen, setIsInviteModalOpen] = useState(false);
  const [loadingMember, setLoadingMember] = useState<string | null>(null);

  useEffect(() => {
    if (currentTenant?.id) {
      loadMembers();
    }
  }, [currentTenant?.id]);

  const loadMembers = async () => {
    if (!currentTenant?.id) return;
    const data = await get(`tenants/${currentTenant.id}/members`);
    if (data) {
      setMembers(Array.isArray(data) ? data : data.members || []);
    }
  };

  const handleRoleChange = async (memberId: string, newRole: string) => {
    if (!currentTenant?.id) return;
    setLoadingMember(memberId);
    try {
      // In a real implementation, this would call an API to update the role
      // await patch(`tenants/${currentTenant.id}/members/${memberId}`, { role: newRole });
      await loadMembers();
    } catch (err) {
      console.error('Failed to update role:', err);
    } finally {
      setLoadingMember(null);
    }
  };

  const handleRemoveMember = async (memberId: string) => {
    if (!currentTenant?.id) return;
    if (!confirm('Are you sure you want to remove this team member?')) return;

    setLoadingMember(memberId);
    try {
      // In a real implementation, this would call an API to remove the member
      // await del(`tenants/${currentTenant.id}/members/${memberId}`);
      await loadMembers();
    } catch (err) {
      console.error('Failed to remove member:', err);
    } finally {
      setLoadingMember(null);
    }
  };

  const handleInvite = async (email: string, role: string) => {
    if (!currentTenant?.id) return;
    const result = await post(`tenants/${currentTenant.id}/members`, {
      email,
      role,
    });
    if (result) {
      setIsInviteModalOpen(false);
      await loadMembers();
    }
  };

  const getMemberName = (member: TeamMember) => {
    if (member.user) {
      const fullName = member.user.full_name ||
        `${member.user.first_name || ''} ${member.user.last_name || ''}`.trim();
      return fullName || member.user.email;
    }
    return member.invited_email || 'Unknown';
  };

  const getMemberEmail = (member: TeamMember) => {
    return member.user?.email || member.invited_email || '';
  };

  const getMemberStatus = (member: TeamMember): 'active' | 'pending' => {
    return member.accepted_at ? 'active' : 'pending';
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Team</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage your team members and their permissions.
          </p>
        </div>
        {canInviteMembers && (
          <button
            onClick={() => setIsInviteModalOpen(true)}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            Invite member
          </button>
        )}
      </div>

      {error && (
        <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700">
          {error.message}
        </div>
      )}

      {/* Team Members Table */}
      <div className="mt-8 overflow-hidden rounded-lg bg-white shadow">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Member
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Role
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                Joined
              </th>
              <th className="relative px-6 py-3">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {isLoading ? (
              <tr>
                <td colSpan={5} className="px-6 py-4 text-center text-gray-500">
                  Loading...
                </td>
              </tr>
            ) : members.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-4 text-center text-gray-500">
                  No team members yet. Invite someone to get started.
                </td>
              </tr>
            ) : (
              members.map((member) => {
                const memberStatus = getMemberStatus(member);
                const memberName = getMemberName(member);
                const memberEmail = getMemberEmail(member);
                const isOwner = member.role === 'owner';

                return (
                  <tr key={member.id}>
                    <td className="whitespace-nowrap px-6 py-4">
                      <div className="flex items-center">
                        <div className="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center text-sm font-medium text-gray-600">
                          {memberName[0]?.toUpperCase() || '?'}
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">{memberName}</div>
                          <div className="text-sm text-gray-500">{memberEmail}</div>
                        </div>
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-6 py-4">
                      {canManageTeam && !isOwner ? (
                        <select
                          value={member.role}
                          onChange={(e) => handleRoleChange(member.id, e.target.value)}
                          disabled={loadingMember === member.id}
                          className="rounded border-gray-300 text-sm"
                        >
                          <option value="admin">Admin</option>
                          <option value="member">Member</option>
                          <option value="viewer">Viewer</option>
                        </select>
                      ) : (
                        <span className="text-sm text-gray-900 capitalize">{member.role}</span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-6 py-4">
                      {memberStatus === 'active' ? (
                        <span className="inline-flex rounded-full bg-green-100 px-2 text-xs font-semibold leading-5 text-green-800">
                          Active
                        </span>
                      ) : (
                        <span className="inline-flex rounded-full bg-yellow-100 px-2 text-xs font-semibold leading-5 text-yellow-800">
                          Pending
                        </span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                      {memberStatus === 'active'
                        ? formatDate(member.accepted_at)
                        : `Invited ${formatDate(member.invited_at)}`}
                    </td>
                    <td className="whitespace-nowrap px-6 py-4 text-right text-sm font-medium">
                      {canManageTeam && !isOwner && (
                        <button
                          onClick={() => handleRemoveMember(member.id)}
                          disabled={loadingMember === member.id}
                          className="text-red-600 hover:text-red-900 disabled:opacity-50"
                        >
                          {loadingMember === member.id ? 'Removing...' : 'Remove'}
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Invite Modal */}
      {isInviteModalOpen && (
        <InviteModal
          onClose={() => setIsInviteModalOpen(false)}
          onInvite={handleInvite}
        />
      )}
    </div>
  );
}

function InviteModal({
  onClose,
  onInvite,
}: {
  onClose: () => void;
  onInvite: (email: string, role: string) => Promise<void>;
}) {
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('member');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await onInvite(email, role);
    } catch (err) {
      setError('Failed to send invitation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
        <h2 className="text-lg font-medium text-gray-900">Invite team member</h2>

        {error && (
          <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email address
            </label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="colleague@company.com"
            />
          </div>
          <div className="mt-4">
            <label htmlFor="role" className="block text-sm font-medium text-gray-700">
              Role
            </label>
            <select
              id="role"
              value={role}
              onChange={(e) => setRole(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="admin">Admin - Can manage team and settings</option>
              <option value="member">Member - Can view and edit content</option>
              <option value="viewer">Viewer - Read-only access</option>
            </select>
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
              disabled={loading || !email}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Sending...' : 'Send invitation'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

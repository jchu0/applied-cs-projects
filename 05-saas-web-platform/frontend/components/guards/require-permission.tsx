'use client';

import { usePermission, useTenant } from '@/lib/hooks';

interface RequirePermissionProps {
  children: React.ReactNode;
  permission: 'canManageTeam' | 'canManageBilling' | 'canDeleteTenant' | 'canInviteMembers' | 'canEdit';
  fallback?: React.ReactNode;
}

export function RequirePermission({ children, permission, fallback }: RequirePermissionProps) {
  const permissions = usePermission();
  const { isLoading } = useTenant();

  if (isLoading) {
    return null;
  }

  if (!permissions[permission]) {
    return fallback || (
      <div className="rounded-md bg-yellow-50 p-4">
        <p className="text-sm text-yellow-700">
          You don&apos;t have permission to access this feature.
        </p>
      </div>
    );
  }

  return <>{children}</>;
}

// Role-based guard
interface RequireRoleProps {
  children: React.ReactNode;
  roles: ('owner' | 'admin' | 'member' | 'viewer')[];
  fallback?: React.ReactNode;
}

export function RequireRole({ children, roles, fallback }: RequireRoleProps) {
  const { userRole, isLoading } = useTenant();

  if (isLoading) {
    return null;
  }

  if (!userRole || !roles.includes(userRole as any)) {
    return fallback || (
      <div className="rounded-md bg-yellow-50 p-4">
        <p className="text-sm text-yellow-700">
          You don&apos;t have the required role to access this feature.
        </p>
      </div>
    );
  }

  return <>{children}</>;
}

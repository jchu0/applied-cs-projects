'use client';

import { useEffect, useState } from 'react';
import { useApi } from '@/lib/hooks/use-api';
import { useTenant } from '@/lib/hooks/use-tenant';

interface DashboardStats {
  totalMembers: number;
  activePlans: string;
  recentActivity: Activity[];
}

interface Activity {
  id: string;
  user: string;
  action: string;
  time: string;
}

export default function DashboardPage() {
  const { get, isLoading, error } = useApi();
  const { currentTenant } = useTenant();
  const [stats, setStats] = useState<DashboardStats>({
    totalMembers: 0,
    activePlans: 'Free',
    recentActivity: [],
  });

  useEffect(() => {
    if (currentTenant?.id) {
      loadDashboardData();
    }
  }, [currentTenant?.id]);

  const loadDashboardData = async () => {
    try {
      // Fetch team members count
      const membersData = await get(`tenants/${currentTenant?.id}/members`);
      const memberCount = membersData?.members?.length || 0;

      // Fetch subscription info
      let planName = 'Free';
      try {
        const subData = await get(`billing/tenants/${currentTenant?.id}/subscription`);
        planName = subData?.plan?.name || 'Free';
      } catch {
        // No subscription, default to Free
      }

      setStats({
        totalMembers: memberCount,
        activePlans: planName,
        recentActivity: [], // Activity would come from audit logs if user has access
      });
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
      <p className="mt-1 text-sm text-gray-500">
        Welcome back! Here&apos;s an overview of your workspace.
      </p>

      {error && (
        <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700">
          {error.message}
        </div>
      )}

      {/* Stats */}
      <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Team Members"
          value={isLoading ? '...' : stats.totalMembers.toString()}
          change=""
        />
        <StatCard
          title="Current Plan"
          value={isLoading ? '...' : stats.activePlans}
          change=""
        />
        <StatCard
          title="Workspace"
          value={currentTenant?.name || '...'}
          change=""
        />
        <StatCard
          title="Status"
          value="Active"
          change=""
        />
      </div>

      {/* Quick Actions */}
      <div className="mt-8">
        <h2 className="text-lg font-medium text-gray-900">Quick Actions</h2>
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <QuickActionCard
            title="Invite Team Member"
            description="Add a new member to your workspace"
            href="/dashboard/team"
          />
          <QuickActionCard
            title="Manage Billing"
            description="View plans and invoices"
            href="/dashboard/billing"
          />
          <QuickActionCard
            title="View API Keys"
            description="Manage your API keys"
            href="/dashboard/settings/api-keys"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, change }: { title: string; value: string; change: string }) {
  const isPositive = change.startsWith('+');
  return (
    <div className="rounded-lg bg-white p-6 shadow">
      <p className="text-sm font-medium text-gray-500">{title}</p>
      <p className="mt-2 text-3xl font-semibold text-gray-900">{value}</p>
      <p className={`mt-2 text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
        {change} from last month
      </p>
    </div>
  );
}

function QuickActionCard({ title, description, href }: { title: string; description: string; href: string }) {
  return (
    <a
      href={href}
      className="block rounded-lg bg-white p-6 shadow hover:shadow-md transition-shadow"
    >
      <h3 className="text-sm font-medium text-gray-900">{title}</h3>
      <p className="mt-1 text-sm text-gray-500">{description}</p>
    </a>
  );
}


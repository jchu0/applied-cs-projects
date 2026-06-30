'use client';

import { useEffect, useState } from 'react';
import { useApi } from '@/lib/hooks/use-api';

interface AdminStats {
  users: {
    total: number;
    active: number;
    new_last_7_days: number;
  };
  tenants: {
    total: number;
    active: number;
  };
  subscriptions: {
    total: number;
    active: number;
    trialing: number;
  };
  revenue: {
    total: number;
    mrr: number;
  };
}

interface GrowthData {
  user_signups: { date: string; count: number }[];
  tenant_creations: { date: string; count: number }[];
}

export default function AdminDashboardPage() {
  const { get, isLoading, error } = useApi();
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [growth, setGrowth] = useState<GrowthData | null>(null);

  useEffect(() => {
    loadAdminData();
  }, []);

  const loadAdminData = async () => {
    try {
      const [statsData, growthData] = await Promise.all([
        get('admin/stats'),
        get('admin/growth-chart?days=30'),
      ]);
      setStats(statsData);
      setGrowth(growthData);
    } catch (err) {
      console.error('Failed to load admin data:', err);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount / 100);
  };

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Admin Overview</h1>
      <p className="mt-1 text-sm text-gray-500">
        Platform-wide metrics and statistics
      </p>

      {error && (
        <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700">
          {error.message}
        </div>
      )}

      {/* Stats Cards */}
      <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Users"
          value={isLoading ? '...' : stats?.users.total.toString() || '0'}
          subtitle={`${stats?.users.active || 0} active`}
          trend={`+${stats?.users.new_last_7_days || 0} this week`}
        />
        <StatCard
          title="Total Tenants"
          value={isLoading ? '...' : stats?.tenants.total.toString() || '0'}
          subtitle={`${stats?.tenants.active || 0} active`}
        />
        <StatCard
          title="Active Subscriptions"
          value={isLoading ? '...' : stats?.subscriptions.active.toString() || '0'}
          subtitle={`${stats?.subscriptions.trialing || 0} trialing`}
        />
        <StatCard
          title="Monthly Revenue"
          value={isLoading ? '...' : formatCurrency(stats?.revenue.mrr || 0)}
          subtitle={`${formatCurrency(stats?.revenue.total || 0)} total`}
        />
      </div>

      {/* Growth Chart Section */}
      <div className="mt-8">
        <h2 className="text-lg font-medium text-gray-900">Growth (Last 30 Days)</h2>
        <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-4">User Signups</h3>
            <div className="h-48 flex items-end space-x-1">
              {growth?.user_signups.slice(-14).map((day, idx) => (
                <div
                  key={idx}
                  className="flex-1 bg-blue-500 rounded-t"
                  style={{
                    height: `${Math.max(10, (day.count / Math.max(...growth.user_signups.map(d => d.count || 1))) * 100)}%`,
                  }}
                  title={`${day.date}: ${day.count}`}
                />
              ))}
            </div>
            <p className="text-xs text-gray-400 mt-2">Last 14 days</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-sm font-medium text-gray-500 mb-4">Tenant Creations</h3>
            <div className="h-48 flex items-end space-x-1">
              {growth?.tenant_creations.slice(-14).map((day, idx) => (
                <div
                  key={idx}
                  className="flex-1 bg-green-500 rounded-t"
                  style={{
                    height: `${Math.max(10, (day.count / Math.max(...growth.tenant_creations.map(d => d.count || 1))) * 100)}%`,
                  }}
                  title={`${day.date}: ${day.count}`}
                />
              ))}
            </div>
            <p className="text-xs text-gray-400 mt-2">Last 14 days</p>
          </div>
        </div>
      </div>

      {/* Quick Links */}
      <div className="mt-8">
        <h2 className="text-lg font-medium text-gray-900">Quick Actions</h2>
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <QuickLink href="/admin/users" title="Manage Users" description="View and manage user accounts" />
          <QuickLink href="/admin/tenants" title="Manage Tenants" description="View and manage organizations" />
          <QuickLink href="/admin/audit-logs" title="Audit Logs" description="View system activity logs" />
          <QuickLink href="/dashboard" title="User Dashboard" description="Return to user dashboard" />
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  trend,
}: {
  title: string;
  value: string;
  subtitle?: string;
  trend?: string;
}) {
  return (
    <div className="rounded-lg bg-white p-6 shadow">
      <p className="text-sm font-medium text-gray-500">{title}</p>
      <p className="mt-2 text-3xl font-semibold text-gray-900">{value}</p>
      {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
      {trend && <p className="mt-1 text-sm text-green-600">{trend}</p>}
    </div>
  );
}

function QuickLink({
  href,
  title,
  description,
}: {
  href: string;
  title: string;
  description: string;
}) {
  return (
    <a
      href={href}
      className="block rounded-lg bg-white p-4 shadow hover:shadow-md transition-shadow"
    >
      <h3 className="text-sm font-medium text-gray-900">{title}</h3>
      <p className="mt-1 text-sm text-gray-500">{description}</p>
    </a>
  );
}

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { RequireAuth } from '@/components/guards/require-auth';

const adminNavItems = [
  { href: '/admin', label: 'Overview', icon: '📊' },
  { href: '/admin/users', label: 'Users', icon: '👥' },
  { href: '/admin/tenants', label: 'Tenants', icon: '🏢' },
  { href: '/admin/audit-logs', label: 'Audit Logs', icon: '📋' },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <RequireAuth requireAdmin>
      <div className="flex h-screen bg-gray-100">
        {/* Admin Sidebar */}
        <aside className="w-64 bg-gray-900 text-white">
          <div className="p-4 border-b border-gray-700">
            <h1 className="text-xl font-bold">Admin Panel</h1>
            <p className="text-sm text-gray-400">Platform Management</p>
          </div>
          <nav className="p-4">
            <ul className="space-y-2">
              {adminNavItems.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      className={`flex items-center px-4 py-2 rounded-md transition-colors ${
                        isActive
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-300 hover:bg-gray-800'
                      }`}
                    >
                      <span className="mr-3">{item.icon}</span>
                      {item.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
          <div className="absolute bottom-0 w-64 p-4 border-t border-gray-700">
            <Link
              href="/dashboard"
              className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-md"
            >
              <span className="mr-3">←</span>
              Back to Dashboard
            </Link>
          </div>
        </aside>

        {/* Main Content */}
        <div className="flex-1 overflow-auto">
          <header className="bg-white shadow-sm border-b">
            <div className="px-6 py-4">
              <h2 className="text-lg font-semibold text-gray-800">Administration</h2>
            </div>
          </header>
          <main className="p-6">
            {children}
          </main>
        </div>
      </div>
    </RequireAuth>
  );
}

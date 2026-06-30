'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

const settingsSections = [
  { name: 'General', href: '/dashboard/settings', description: 'Basic workspace settings' },
  { name: 'API Keys', href: '/dashboard/settings/api-keys', description: 'Manage API access' },
  { name: 'Notifications', href: '/dashboard/settings/notifications', description: 'Email and notification preferences' },
  { name: 'Integrations', href: '/dashboard/settings/integrations', description: 'Connect external services' },
];

export default function SettingsPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
      <p className="mt-1 text-sm text-gray-500">
        Manage your workspace settings and preferences.
      </p>

      <div className="mt-8 grid gap-6 lg:grid-cols-4">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <nav className="space-y-1">
            {settingsSections.map((section) => (
              <Link
                key={section.name}
                href={section.href}
                className="block rounded-md px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100"
              >
                {section.name}
              </Link>
            ))}
          </nav>
        </div>

        {/* General Settings Form */}
        <div className="lg:col-span-3">
          <div className="rounded-lg bg-white p-6 shadow">
            <h2 className="text-lg font-medium text-gray-900">General Settings</h2>

            <form className="mt-6 space-y-6">
              <div>
                <label htmlFor="workspaceName" className="block text-sm font-medium text-gray-700">
                  Workspace name
                </label>
                <input
                  type="text"
                  id="workspaceName"
                  defaultValue="My Workspace"
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>

              <div>
                <label htmlFor="workspaceSlug" className="block text-sm font-medium text-gray-700">
                  Workspace URL
                </label>
                <div className="mt-1 flex rounded-md shadow-sm">
                  <span className="inline-flex items-center rounded-l-md border border-r-0 border-gray-300 bg-gray-50 px-3 text-gray-500 sm:text-sm">
                    app.saas.com/
                  </span>
                  <input
                    type="text"
                    id="workspaceSlug"
                    defaultValue="my-workspace"
                    className="block w-full rounded-none rounded-r-md border border-gray-300 px-3 py-2 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="timezone" className="block text-sm font-medium text-gray-700">
                  Timezone
                </label>
                <select
                  id="timezone"
                  defaultValue="America/New_York"
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  <option value="America/New_York">Eastern Time (US & Canada)</option>
                  <option value="America/Chicago">Central Time (US & Canada)</option>
                  <option value="America/Denver">Mountain Time (US & Canada)</option>
                  <option value="America/Los_Angeles">Pacific Time (US & Canada)</option>
                  <option value="Europe/London">London</option>
                  <option value="Europe/Paris">Paris</option>
                  <option value="Asia/Tokyo">Tokyo</option>
                </select>
              </div>

              <div>
                <label htmlFor="language" className="block text-sm font-medium text-gray-700">
                  Language
                </label>
                <select
                  id="language"
                  defaultValue="en"
                  className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                >
                  <option value="en">English</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                  <option value="ja">Japanese</option>
                </select>
              </div>

              <div className="flex justify-end">
                <button
                  type="submit"
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
                >
                  Save changes
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

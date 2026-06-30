'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/lib/hooks/use-auth';
import { useApi } from '@/lib/hooks/use-api';

export default function ProfilePage() {
  const { user, updateProfile } = useAuth();
  const { post, isLoading: apiLoading, error } = useApi();
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [profile, setProfile] = useState({
    firstName: '',
    lastName: '',
    email: '',
  });

  // Password change state
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordSuccess, setPasswordSuccess] = useState<string | null>(null);
  const [changingPassword, setChangingPassword] = useState(false);

  useEffect(() => {
    if (user) {
      setProfile({
        firstName: user.firstName || '',
        lastName: user.lastName || '',
        email: user.email || '',
      });
    }
  }, [user]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setSuccess(null);

    try {
      await updateProfile({
        firstName: profile.firstName,
        lastName: profile.lastName,
      });
      setSuccess('Profile updated successfully');
    } catch (err) {
      // Error handled by useAuth
    } finally {
      setIsLoading(false);
    }
  };

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordError(null);
    setPasswordSuccess(null);

    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      setPasswordError('Passwords do not match');
      return;
    }

    if (passwordForm.newPassword.length < 8) {
      setPasswordError('Password must be at least 8 characters');
      return;
    }

    setChangingPassword(true);
    try {
      const result = await post('auth/password/change', {
        current_password: passwordForm.currentPassword,
        new_password: passwordForm.newPassword,
      });

      if (result) {
        setPasswordSuccess('Password changed successfully');
        setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
      }
    } catch (err) {
      setPasswordError('Failed to change password');
    } finally {
      setChangingPassword(false);
    }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Profile</h1>
      <p className="mt-1 text-sm text-gray-500">
        Manage your personal information and account settings.
      </p>

      {error && (
        <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700">
          {error.message}
        </div>
      )}

      <div className="mt-8 max-w-2xl">
        {/* Avatar Section */}
        <div className="rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-medium text-gray-900">Profile Photo</h2>
          <div className="mt-4 flex items-center">
            <div className="h-20 w-20 rounded-full bg-gray-300 flex items-center justify-center text-2xl text-gray-600">
              {user?.firstName?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase() || '?'}
            </div>
            <div className="ml-6">
              <button className="rounded-md bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm ring-1 ring-gray-300 hover:bg-gray-50">
                Change photo
              </button>
              <button className="ml-3 text-sm font-medium text-gray-600 hover:text-gray-900">
                Remove
              </button>
            </div>
          </div>
        </div>

        {/* Profile Form */}
        <form onSubmit={handleSubmit} className="mt-6 rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-medium text-gray-900">Personal Information</h2>

          {success && (
            <div className="mt-4 rounded-md bg-green-50 p-4 text-sm text-green-700">
              {success}
            </div>
          )}

          <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="firstName" className="block text-sm font-medium text-gray-700">
                First name
              </label>
              <input
                type="text"
                id="firstName"
                value={profile.firstName}
                onChange={(e) => setProfile({ ...profile, firstName: e.target.value })}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <div>
              <label htmlFor="lastName" className="block text-sm font-medium text-gray-700">
                Last name
              </label>
              <input
                type="text"
                id="lastName"
                value={profile.lastName}
                onChange={(e) => setProfile({ ...profile, lastName: e.target.value })}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <div className="sm:col-span-2">
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Email address
              </label>
              <input
                type="email"
                id="email"
                value={profile.email}
                disabled
                className="mt-1 block w-full rounded-md border border-gray-300 bg-gray-50 px-3 py-2 text-gray-500 shadow-sm"
              />
              <p className="mt-1 text-xs text-gray-500">
                Contact support to change your email address.
              </p>
            </div>
          </div>
          <div className="mt-6 flex justify-end">
            <button
              type="submit"
              disabled={isLoading}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? 'Saving...' : 'Save changes'}
            </button>
          </div>
        </form>

        {/* Security Section */}
        <form onSubmit={handlePasswordChange} className="mt-6 rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-medium text-gray-900">Change Password</h2>

          {passwordError && (
            <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700">
              {passwordError}
            </div>
          )}

          {passwordSuccess && (
            <div className="mt-4 rounded-md bg-green-50 p-4 text-sm text-green-700">
              {passwordSuccess}
            </div>
          )}

          <div className="mt-4 space-y-4">
            <div>
              <label htmlFor="currentPassword" className="block text-sm font-medium text-gray-700">
                Current password
              </label>
              <input
                type="password"
                id="currentPassword"
                value={passwordForm.currentPassword}
                onChange={(e) => setPasswordForm({ ...passwordForm, currentPassword: e.target.value })}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <div>
              <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700">
                New password
              </label>
              <input
                type="password"
                id="newPassword"
                value={passwordForm.newPassword}
                onChange={(e) => setPasswordForm({ ...passwordForm, newPassword: e.target.value })}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                Confirm new password
              </label>
              <input
                type="password"
                id="confirmPassword"
                value={passwordForm.confirmPassword}
                onChange={(e) => setPasswordForm({ ...passwordForm, confirmPassword: e.target.value })}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
          </div>
          <div className="mt-6 flex justify-end">
            <button
              type="submit"
              disabled={changingPassword || !passwordForm.currentPassword || !passwordForm.newPassword}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {changingPassword ? 'Changing...' : 'Change password'}
            </button>
          </div>
        </form>

        {/* Two-Factor Authentication */}
        <div className="mt-6 rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-medium text-gray-900">Two-Factor Authentication</h2>
          <p className="mt-2 text-sm text-gray-500">
            Add an extra layer of security to your account by enabling two-factor authentication.
          </p>
          <button className="mt-4 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50">
            Enable 2FA
          </button>
        </div>

        {/* Danger Zone */}
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 p-6">
          <h2 className="text-lg font-medium text-red-900">Danger Zone</h2>
          <p className="mt-2 text-sm text-red-700">
            Once you delete your account, there is no going back. Please be certain.
          </p>
          <button className="mt-4 rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700">
            Delete account
          </button>
        </div>
      </div>
    </div>
  );
}

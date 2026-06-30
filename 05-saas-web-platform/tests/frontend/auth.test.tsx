/**
 * Test suite for authentication components and hooks.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Components
import LoginForm from '@/components/auth/LoginForm';
import RegisterForm from '@/components/auth/RegisterForm';
import PasswordResetForm from '@/components/auth/PasswordResetForm';
import AuthGuard from '@/components/auth/AuthGuard';
import OAuthButtons from '@/components/auth/OAuthButtons';

// Hooks
import { useAuth } from '@/hooks/useAuth';
import { useSession } from '@/hooks/useSession';

// Services
import * as authService from '@/services/auth';

// Mock services
jest.mock('@/services/auth');

// Test utilities
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

const createTestStore = () =>
  configureStore({
    reducer: {
      auth: (state = { user: null, isAuthenticated: false }) => state,
    },
  });

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = createTestQueryClient();
  const store = createTestStore();

  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>{children}</BrowserRouter>
      </QueryClientProvider>
    </Provider>
  );
};

describe('LoginForm', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders login form with all fields', () => {
    render(
      <TestWrapper>
        <LoginForm />
      </TestWrapper>
    );

    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    expect(screen.getByText(/forgot password/i)).toBeInTheDocument();
  });

  test('validates email format', async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <LoginForm />
      </TestWrapper>
    );

    const emailInput = screen.getByLabelText(/email/i);
    const submitButton = screen.getByRole('button', { name: /sign in/i });

    await user.type(emailInput, 'invalid-email');
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/please enter a valid email/i)).toBeInTheDocument();
    });
  });

  test('submits login form with valid data', async () => {
    const mockLogin = authService.login as jest.MockedFunction<typeof authService.login>;
    mockLogin.mockResolvedValue({
      access_token: 'test-token',
      refresh_token: 'refresh-token',
      user: {
        id: 1,
        email: 'test@example.com',
        username: 'testuser',
      },
    });

    const user = userEvent.setup();

    render(
      <TestWrapper>
        <LoginForm />
      </TestWrapper>
    );

    const emailInput = screen.getByLabelText(/email/i);
    const passwordInput = screen.getByLabelText(/password/i);
    const submitButton = screen.getByRole('button', { name: /sign in/i });

    await user.type(emailInput, 'test@example.com');
    await user.type(passwordInput, 'TestPassword123!');
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'TestPassword123!',
      });
    });
  });

  test('displays error message on failed login', async () => {
    const mockLogin = authService.login as jest.MockedFunction<typeof authService.login>;
    mockLogin.mockRejectedValue(new Error('Invalid credentials'));

    const user = userEvent.setup();

    render(
      <TestWrapper>
        <LoginForm />
      </TestWrapper>
    );

    const emailInput = screen.getByLabelText(/email/i);
    const passwordInput = screen.getByLabelText(/password/i);
    const submitButton = screen.getByRole('button', { name: /sign in/i });

    await user.type(emailInput, 'test@example.com');
    await user.type(passwordInput, 'WrongPassword');
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });

  test('shows loading state during submission', async () => {
    const mockLogin = authService.login as jest.MockedFunction<typeof authService.login>;
    mockLogin.mockImplementation(
      () =>
        new Promise((resolve) =>
          setTimeout(() => resolve({ access_token: 'token' }), 1000)
        )
    );

    const user = userEvent.setup();

    render(
      <TestWrapper>
        <LoginForm />
      </TestWrapper>
    );

    const submitButton = screen.getByRole('button', { name: /sign in/i });

    await user.type(screen.getByLabelText(/email/i), 'test@example.com');
    await user.type(screen.getByLabelText(/password/i), 'TestPassword123!');
    await user.click(submitButton);

    expect(screen.getByText(/signing in/i)).toBeInTheDocument();
    expect(submitButton).toBeDisabled();
  });
});

describe('RegisterForm', () => {
  test('renders registration form with all fields', () => {
    render(
      <TestWrapper>
        <RegisterForm />
      </TestWrapper>
    );

    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^password/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
  });

  test('validates password strength', async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <RegisterForm />
      </TestWrapper>
    );

    const passwordInput = screen.getByLabelText(/^password/i);

    await user.type(passwordInput, 'weak');

    await waitFor(() => {
      expect(
        screen.getByText(/password must be at least 8 characters/i)
      ).toBeInTheDocument();
    });
  });

  test('validates password confirmation match', async () => {
    const user = userEvent.setup();

    render(
      <TestWrapper>
        <RegisterForm />
      </TestWrapper>
    );

    const passwordInput = screen.getByLabelText(/^password/i);
    const confirmInput = screen.getByLabelText(/confirm password/i);

    await user.type(passwordInput, 'StrongPassword123!');
    await user.type(confirmInput, 'DifferentPassword123!');
    await user.tab(); // Trigger validation

    await waitFor(() => {
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
    });
  });

  test('submits registration form with valid data', async () => {
    const mockRegister = authService.register as jest.MockedFunction<
      typeof authService.register
    >;
    mockRegister.mockResolvedValue({
      access_token: 'test-token',
      user: {
        id: 1,
        username: 'newuser',
        email: 'new@example.com',
      },
    });

    const user = userEvent.setup();

    render(
      <TestWrapper>
        <RegisterForm />
      </TestWrapper>
    );

    await user.type(screen.getByLabelText(/username/i), 'newuser');
    await user.type(screen.getByLabelText(/email/i), 'new@example.com');
    await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
    await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
    await user.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith({
        username: 'newuser',
        email: 'new@example.com',
        password: 'StrongPassword123!',
        password_confirm: 'StrongPassword123!',
      });
    });
  });

  test('checks username availability', async () => {
    const mockCheckUsername = authService.checkUsernameAvailability as jest.MockedFunction<
      typeof authService.checkUsernameAvailability
    >;
    mockCheckUsername.mockResolvedValue({ available: false });

    const user = userEvent.setup();

    render(
      <TestWrapper>
        <RegisterForm />
      </TestWrapper>
    );

    const usernameInput = screen.getByLabelText(/username/i);

    await user.type(usernameInput, 'existinguser');
    await user.tab();

    await waitFor(() => {
      expect(screen.getByText(/username is already taken/i)).toBeInTheDocument();
    });
  });
});

describe('PasswordResetForm', () => {
  test('renders password reset form', () => {
    render(
      <TestWrapper>
        <PasswordResetForm />
      </TestWrapper>
    );

    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /send reset link/i })
    ).toBeInTheDocument();
  });

  test('submits password reset request', async () => {
    const mockResetPassword = authService.requestPasswordReset as jest.MockedFunction<
      typeof authService.requestPasswordReset
    >;
    mockResetPassword.mockResolvedValue({ success: true });

    const user = userEvent.setup();

    render(
      <TestWrapper>
        <PasswordResetForm />
      </TestWrapper>
    );

    await user.type(screen.getByLabelText(/email/i), 'forgot@example.com');
    await user.click(screen.getByRole('button', { name: /send reset link/i }));

    await waitFor(() => {
      expect(mockResetPassword).toHaveBeenCalledWith('forgot@example.com');
      expect(
        screen.getByText(/reset link sent to your email/i)
      ).toBeInTheDocument();
    });
  });
});

describe('AuthGuard', () => {
  test('renders children when authenticated', () => {
    const store = configureStore({
      reducer: {
        auth: () => ({
          user: { id: 1, email: 'test@example.com' },
          isAuthenticated: true,
        }),
      },
    });

    render(
      <Provider store={store}>
        <QueryClientProvider client={createTestQueryClient()}>
          <BrowserRouter>
            <AuthGuard>
              <div>Protected Content</div>
            </AuthGuard>
          </BrowserRouter>
        </QueryClientProvider>
      </Provider>
    );

    expect(screen.getByText('Protected Content')).toBeInTheDocument();
  });

  test('redirects to login when not authenticated', () => {
    const store = configureStore({
      reducer: {
        auth: () => ({
          user: null,
          isAuthenticated: false,
        }),
      },
    });

    render(
      <Provider store={store}>
        <QueryClientProvider client={createTestQueryClient()}>
          <BrowserRouter>
            <AuthGuard>
              <div>Protected Content</div>
            </AuthGuard>
          </BrowserRouter>
        </QueryClientProvider>
      </Provider>
    );

    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
  });

  test('checks role-based access', () => {
    const store = configureStore({
      reducer: {
        auth: () => ({
          user: { id: 1, email: 'test@example.com', role: 'user' },
          isAuthenticated: true,
        }),
      },
    });

    render(
      <Provider store={store}>
        <QueryClientProvider client={createTestQueryClient()}>
          <BrowserRouter>
            <AuthGuard requiredRole="admin">
              <div>Admin Content</div>
            </AuthGuard>
          </BrowserRouter>
        </QueryClientProvider>
      </Provider>
    );

    expect(screen.queryByText('Admin Content')).not.toBeInTheDocument();
    expect(screen.getByText(/access denied/i)).toBeInTheDocument();
  });
});

describe('OAuthButtons', () => {
  test('renders OAuth provider buttons', () => {
    render(
      <TestWrapper>
        <OAuthButtons />
      </TestWrapper>
    );

    expect(screen.getByRole('button', { name: /google/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /github/i })).toBeInTheDocument();
  });

  test('handles Google OAuth login', async () => {
    const mockGoogleLogin = authService.oauthLogin as jest.MockedFunction<
      typeof authService.oauthLogin
    >;
    mockGoogleLogin.mockResolvedValue({
      access_token: 'google-token',
      user: { id: 1, email: 'google@example.com' },
    });

    const user = userEvent.setup();

    render(
      <TestWrapper>
        <OAuthButtons />
      </TestWrapper>
    );

    await user.click(screen.getByRole('button', { name: /google/i }));

    // Google OAuth flow would be initiated
    await waitFor(() => {
      expect(window.location.href).toContain('accounts.google.com');
    });
  });

  test('handles GitHub OAuth login', async () => {
    const mockGithubLogin = authService.oauthLogin as jest.MockedFunction<
      typeof authService.oauthLogin
    >;
    mockGithubLogin.mockResolvedValue({
      access_token: 'github-token',
      user: { id: 1, email: 'github@example.com' },
    });

    const user = userEvent.setup();

    render(
      <TestWrapper>
        <OAuthButtons />
      </TestWrapper>
    );

    await user.click(screen.getByRole('button', { name: /github/i }));

    // GitHub OAuth flow would be initiated
    await waitFor(() => {
      expect(window.location.href).toContain('github.com/login/oauth');
    });
  });
});

describe('useAuth Hook', () => {
  test('provides authentication state and methods', () => {
    const { result } = renderHook(() => useAuth(), {
      wrapper: TestWrapper,
    });

    expect(result.current).toHaveProperty('user');
    expect(result.current).toHaveProperty('isAuthenticated');
    expect(result.current).toHaveProperty('login');
    expect(result.current).toHaveProperty('logout');
    expect(result.current).toHaveProperty('register');
  });

  test('handles login flow', async () => {
    const mockLogin = authService.login as jest.MockedFunction<typeof authService.login>;
    mockLogin.mockResolvedValue({
      access_token: 'test-token',
      user: { id: 1, email: 'test@example.com' },
    });

    const { result } = renderHook(() => useAuth(), {
      wrapper: TestWrapper,
    });

    await act(async () => {
      await result.current.login('test@example.com', 'password');
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).toEqual({
      id: 1,
      email: 'test@example.com',
    });
  });

  test('handles logout flow', async () => {
    const { result } = renderHook(() => useAuth(), {
      wrapper: TestWrapper,
    });

    await act(async () => {
      await result.current.logout();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBe(null);
  });
});

describe('useSession Hook', () => {
  test('manages session lifecycle', () => {
    const { result } = renderHook(() => useSession(), {
      wrapper: TestWrapper,
    });

    expect(result.current).toHaveProperty('session');
    expect(result.current).toHaveProperty('isActive');
    expect(result.current).toHaveProperty('extend');
    expect(result.current).toHaveProperty('expire');
  });

  test('auto-refreshes token before expiry', async () => {
    const mockRefreshToken = authService.refreshToken as jest.MockedFunction<
      typeof authService.refreshToken
    >;
    mockRefreshToken.mockResolvedValue({
      access_token: 'new-token',
    });

    const { result } = renderHook(() => useSession(), {
      wrapper: TestWrapper,
    });

    // Simulate token near expiry
    jest.advanceTimersByTime(25 * 60 * 1000); // 25 minutes

    await waitFor(() => {
      expect(mockRefreshToken).toHaveBeenCalled();
    });
  });

  test('handles session timeout', async () => {
    const { result } = renderHook(() => useSession(), {
      wrapper: TestWrapper,
    });

    // Simulate inactivity timeout
    jest.advanceTimersByTime(30 * 60 * 1000); // 30 minutes

    await waitFor(() => {
      expect(result.current.isActive).toBe(false);
      expect(screen.getByText(/session expired/i)).toBeInTheDocument();
    });
  });
});
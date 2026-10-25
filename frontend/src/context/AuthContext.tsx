import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { env } from '@/config/env';
import { apiClient, authService } from '@/services';
import type { AuthStatus, LoginCredentials, RegisterPayload, User } from '@/types/auth';
import { AuthContext, type AuthContextValue } from './auth-context';

/**
 * Centralised authentication state.
 *
 * - The session is restored from the backend (GET /auth/me) on load.
 * - No passwords or tokens are kept in browser storage. The backend is
 *   expected to manage the session in an HTTP-only cookie.
 * - Client-side route protection is a convenience only; the server must
 *   enforce authorization on every request.
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('loading');
  const [user, setUser] = useState<User | null>(null);
  const [sessionExpired, setSessionExpired] = useState(false);
  const userRef = useRef<User | null>(null);

  const applyUser = useCallback((next: User | null) => {
    userRef.current = next;
    setUser(next);
    setStatus(next ? 'authenticated' : 'unauthenticated');
  }, []);

  // Restore the session from the backend.
  useEffect(() => {
    const controller = new AbortController();
    authService
      .me(controller.signal)
      .then((restored) => {
        if (!controller.signal.aborted) applyUser(restored);
      })
      .catch(() => {
        if (!controller.signal.aborted) applyUser(null);
      });
    return () => controller.abort();
  }, [applyUser]);

  // A 401 on an authenticated request means the session expired.
  useEffect(() => {
    apiClient.setUnauthorizedHandler(() => {
      if (userRef.current) {
        apiClient.setToken(null);
        applyUser(null);
        setSessionExpired(true);
      }
    });
    return () => apiClient.setUnauthorizedHandler(null);
  }, [applyUser]);

  const login = useCallback(
    async (credentials: LoginCredentials) => {
      const signedIn = await authService.login(credentials);
      setSessionExpired(false);
      applyUser(signedIn);
      return signedIn;
    },
    [applyUser],
  );

  const register = useCallback((payload: RegisterPayload) => authService.register(payload), []);

  const logout = useCallback(async () => {
    try {
      await authService.logout();
    } catch {
      // The local session is cleared regardless; the server cookie expires on its own.
    }
    setSessionExpired(false);
    applyUser(null);
  }, [applyUser]);

  const clearSessionExpired = useCallback(() => setSessionExpired(false), []);

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      user,
      sessionExpired,
      isDemoAuth: env.demoMode,
      login,
      register,
      logout,
      clearSessionExpired,
    }),
    [status, user, sessionExpired, login, register, logout, clearSessionExpired],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

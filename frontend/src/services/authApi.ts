import { asRecord, asString, pick } from '@/lib/safe';
import type { LoginCredentials, RegisterPayload, RegisterResult, User } from '@/types/auth';
import { apiClient } from './apiClient';
import { ApiError, isApiError } from './errors';
import { endpoints } from './endpoints';
import { normalizeUser } from './normalizers';
import type { AuthApi } from './types';

/**
 * Authentication against the real backend.
 *
 * The session is expected to live in a server-managed, HTTP-only cookie, so
 * the browser sends it automatically (credentials: "include"). This module
 * never stores passwords or tokens anywhere in the browser.
 */

async function me(signal?: AbortSignal): Promise<User> {
  const raw = await apiClient.request(endpoints.me, {
    signal,
    suppressUnauthorizedHandler: true,
  });
  const user = normalizeUser(raw);
  if (!user) {
    throw new ApiError({
      kind: 'bad_response',
      message: 'The server did not return a valid user profile.',
    });
  }
  return user;
}

async function register(payload: RegisterPayload): Promise<RegisterResult> {
  const raw = await apiClient.request(endpoints.register, {
    method: 'POST',
    body: {
      full_name: payload.fullName.trim(),
      email: payload.email.trim(),
      password: payload.password,
    },
    suppressUnauthorizedHandler: true,
  });
  const token = asString(pick(asRecord(raw), 'token'));
  if (token) apiClient.setToken(token);
  return {
    user: normalizeUser(raw) ?? undefined,
    message: asString(pick(asRecord(raw), 'message', 'detail')),
  };
}

async function login(credentials: LoginCredentials): Promise<User> {
  const raw = await apiClient.request(endpoints.login, {
    method: 'POST',
    body: {
      email: credentials.email.trim(),
      password: credentials.password,
      ...(credentials.rememberMe === undefined ? {} : { remember_me: credentials.rememberMe }),
    },
    suppressUnauthorizedHandler: true,
  });

  const token = asString(pick(asRecord(raw), 'token'));
  if (token) apiClient.setToken(token);

  // Some backends return the user, others only set the session cookie.
  const direct = normalizeUser(raw);
  if (direct) return direct;

  try {
    return await me();
  } catch (error) {
    if (isApiError(error) && error.kind === 'unauthorized') {
      throw new ApiError({
        kind: 'bad_response',
        status: error.status,
        message:
          'Sign-in was accepted, but the browser did not keep the session. Check the backend CORS and cookie settings (credentials, SameSite, Secure).',
      });
    }
    throw error;
  }
}

async function logout(): Promise<void> {
  apiClient.setToken(null);
  await apiClient.request(endpoints.logout, {
    method: 'POST',
    suppressUnauthorizedHandler: true,
  });
}

export const authApi: AuthApi = { register, login, logout, me };

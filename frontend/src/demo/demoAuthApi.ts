import { sleep } from '@/lib/async';
import { ApiError } from '@/services/errors';
import type { AuthApi } from '@/services/types';
import type { LoginCredentials, RegisterPayload, RegisterResult, User } from '@/types/auth';
import { demoAccounts, demoSession } from './demoStore';

/**
 * DEMO authentication. This is NOT secure and verifies nothing: any
 * well-formed email is accepted. Passwords are never stored. It exists only
 * so the signed-in parts of the interface can be reviewed without a backend.
 */

const LATENCY_MS = 450;

function makeUser(email: string, fullName: string): User {
  return {
    id: `demo-${email.toLowerCase()}`,
    email,
    fullName,
    createdAt: new Date().toISOString(),
    status: 'demo session',
    role: 'Demo user',
    isDemo: true,
  };
}

async function register(payload: RegisterPayload): Promise<RegisterResult> {
  await sleep(LATENCY_MS);
  if (demoAccounts.find(payload.email)) {
    throw new ApiError({
      kind: 'conflict',
      status: 409,
      message: 'A demo account with this email already exists in this browser session.',
      fieldErrors: { email: 'This email is already registered in demo mode.' },
    });
  }
  const user = makeUser(payload.email.trim(), payload.fullName.trim());
  demoAccounts.add(user);
  return { user, message: 'Demo account created for this browser session only.' };
}

async function login(credentials: LoginCredentials): Promise<User> {
  await sleep(LATENCY_MS);
  const email = credentials.email.trim();
  const user = demoAccounts.find(email) ?? makeUser(email, email.split('@')[0] ?? email);
  demoSession.set(user);
  return user;
}

async function logout(): Promise<void> {
  demoSession.clear();
}

async function me(): Promise<User> {
  const user = demoSession.get();
  if (!user) {
    throw new ApiError({ kind: 'unauthorized', status: 401, message: 'No demo session.' });
  }
  return user;
}

export const demoAuthApi: AuthApi = { register, login, logout, me };

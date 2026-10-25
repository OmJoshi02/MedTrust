import { createContext } from 'react';
import type { AuthStatus, LoginCredentials, RegisterPayload, RegisterResult, User } from '@/types/auth';

export interface AuthContextValue {
  status: AuthStatus;
  user: User | null;
  /** True after an authenticated session was rejected by the server (HTTP 401). */
  sessionExpired: boolean;
  /** True when the app runs on the explicitly enabled, non-secure demo flow. */
  isDemoAuth: boolean;
  login: (credentials: LoginCredentials) => Promise<User>;
  register: (payload: RegisterPayload) => Promise<RegisterResult>;
  logout: () => Promise<void>;
  clearSessionExpired: () => void;
}

export const AuthContext = createContext<AuthContextValue | null>(null);

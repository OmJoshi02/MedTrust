export interface User {
  id: string;
  email: string;
  fullName: string;
  createdAt?: string;
  /** Account status as reported by the API, e.g. "active". */
  status?: string;
  role?: string;
  /** True only for the explicitly enabled, non-secure demo session. */
  isDemo: boolean;
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterPayload {
  fullName: string;
  email: string;
  password: string;
}

export interface RegisterResult {
  /** Present only if the API returned the created user. */
  user?: User;
  message?: string;
}

export type AuthStatus = 'loading' | 'authenticated' | 'unauthenticated';

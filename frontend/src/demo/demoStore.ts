import type { AnalysisReport } from '@/types/analysis';
import type { User } from '@/types/auth';

/**
 * Session-scoped storage for demo mode. Nothing here is secure: no passwords
 * are ever stored, only a display profile, and everything lives in
 * sessionStorage so it disappears when the tab is closed.
 */

const SESSION_KEY = 'medtrust.demo.session';
const ACCOUNTS_KEY = 'medtrust.demo.accounts';
const ANALYSES_KEY = 'medtrust.demo.analyses';

function read<T>(key: string): T | null {
  try {
    const raw = window.sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function write(key: string, value: unknown): void {
  try {
    window.sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: demo state simply will not persist */
  }
}

export const demoSession = {
  get(): User | null {
    return read<User>(SESSION_KEY);
  },
  set(user: User): void {
    write(SESSION_KEY, user);
  },
  clear(): void {
    try {
      window.sessionStorage.removeItem(SESSION_KEY);
    } catch {
      /* ignore */
    }
  },
};

export const demoAccounts = {
  all(): User[] {
    return read<User[]>(ACCOUNTS_KEY) ?? [];
  },
  find(email: string): User | undefined {
    const needle = email.trim().toLowerCase();
    return this.all().find((u) => u.email.toLowerCase() === needle);
  },
  add(user: User): void {
    write(ACCOUNTS_KEY, [...this.all(), user]);
  },
};

export const demoAnalyses = {
  all(): AnalysisReport[] {
    return read<AnalysisReport[]>(ANALYSES_KEY) ?? [];
  },
  add(report: AnalysisReport): void {
    write(ANALYSES_KEY, [report, ...this.all()]);
  },
  delete(id: string): void {
    write(ANALYSES_KEY, this.all().filter((r) => r.id !== id));
  },
};

/** Uploaded files for this page session only (cannot be serialised). */
export const demoUploads = new Map<string, Blob>();

/**
 * Central, typed access to build-time configuration.
 * Every setting comes from a VITE_* environment variable (see .env.example).
 */

export interface TeamMember {
  name: string;
  role?: string;
}

function readBool(value: string | undefined, fallback = false): boolean {
  if (value === undefined || value.trim() === '') return fallback;
  return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
}

function readInt(value: string | undefined, fallback: number): number {
  if (value === undefined || value.trim() === '') return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, '');
}

function normalizePrefix(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '');
  if (trimmed === '') return '';
  return trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
}

/** Parses VITE_PROJECT_TEAM. Returns an empty list if unset or malformed. */
export function parseTeam(value: string | undefined): TeamMember[] {
  if (!value || value.trim() === '') return [];
  try {
    const parsed: unknown = JSON.parse(value);
    if (!Array.isArray(parsed)) return [];
    const members: TeamMember[] = [];
    for (const entry of parsed) {
      if (typeof entry === 'object' && entry !== null && 'name' in entry) {
        const { name, role } = entry as { name: unknown; role?: unknown };
        if (typeof name === 'string' && name.trim() !== '') {
          members.push({
            name: name.trim(),
            role: typeof role === 'string' && role.trim() !== '' ? role.trim() : undefined,
          });
        }
      }
    }
    return members;
  } catch {
    return [];
  }
}

const raw = import.meta.env;

export const env = {
  apiBaseUrl:
    raw.VITE_API_BASE_URL !== undefined
      ? trimTrailingSlash(raw.VITE_API_BASE_URL.trim())
      : 'http://127.0.0.1:8000',
  apiPrefix: normalizePrefix(raw.VITE_API_PREFIX ?? '/api/v1'),
  demoMode: readBool(raw.VITE_DEMO_MODE, false),
  requestTimeoutMs: readInt(raw.VITE_REQUEST_TIMEOUT_MS, 15_000),
  analysisTimeoutMs: readInt(raw.VITE_ANALYSIS_TIMEOUT_MS, 120_000),
  maxUploadMb: readInt(raw.VITE_MAX_UPLOAD_MB, 15),
  enableRememberMe: readBool(raw.VITE_ENABLE_REMEMBER_ME, false),
  institution: raw.VITE_INSTITUTION?.trim() || '',
  contactEmail: raw.VITE_CONTACT_EMAIL?.trim() || '',
  team: parseTeam(raw.VITE_PROJECT_TEAM),
} as const;

export type AppEnv = typeof env;

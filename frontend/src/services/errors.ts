export type ApiErrorKind =
  | 'network'
  | 'timeout'
  | 'aborted'
  | 'unauthorized'
  | 'forbidden'
  | 'not_found'
  | 'validation'
  | 'conflict'
  | 'payload_too_large'
  | 'unsupported_media'
  | 'rate_limited'
  | 'server'
  | 'bad_response'
  | 'unknown';

export interface ApiErrorInit {
  kind: ApiErrorKind;
  message: string;
  status?: number;
  /** Per-field messages when the API reports validation errors. */
  fieldErrors?: Record<string, string>;
}

/** Error thrown by every service call. UI code branches on `kind`. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly fieldErrors?: Record<string, string>;

  constructor({ kind, message, status, fieldErrors }: ApiErrorInit) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

export function kindFromStatus(status: number): ApiErrorKind {
  if (status === 401) return 'unauthorized';
  if (status === 403) return 'forbidden';
  if (status === 404) return 'not_found';
  if (status === 409) return 'conflict';
  if (status === 413) return 'payload_too_large';
  if (status === 415) return 'unsupported_media';
  if (status === 429) return 'rate_limited';
  if (status === 400 || status === 422) return 'validation';
  if (status >= 500) return 'server';
  return 'unknown';
}

const DEFAULT_MESSAGES: Record<ApiErrorKind, string> = {
  network: 'The MedTrust service could not be reached. Check your connection and that the backend is running.',
  timeout: 'The request took too long and was cancelled. Please try again.',
  aborted: 'The request was cancelled.',
  unauthorized: 'You are not signed in, or your session has expired.',
  forbidden: 'You do not have permission to access this resource.',
  not_found: 'The requested resource could not be found.',
  validation: 'The server rejected the request. Please check the information provided.',
  conflict: 'This request conflicts with existing data.',
  payload_too_large: 'The file is larger than the server allows.',
  unsupported_media: 'The server does not support this file format.',
  rate_limited: 'Too many requests. Please wait a moment and try again.',
  server: 'The server encountered an error. Please try again later.',
  bad_response: 'The server returned a response this application could not read.',
  unknown: 'Something went wrong. Please try again.',
};

export function defaultMessageFor(kind: ApiErrorKind): string {
  return DEFAULT_MESSAGES[kind];
}

/** A safe, user-facing message for any thrown value. */
export function describeError(error: unknown): string {
  if (isApiError(error)) return error.message || defaultMessageFor(error.kind);
  if (error instanceof Error && error.message) return error.message;
  return defaultMessageFor('unknown');
}

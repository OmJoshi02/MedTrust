import { env } from '@/config/env';
import { isRecord } from '@/lib/safe';
import { ApiError, defaultMessageFor, kindFromStatus } from './errors';

type Primitive = string | number | boolean | null | undefined;

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  /** Serialised as JSON, unless it is a FormData instance. */
  body?: unknown;
  query?: Record<string, Primitive>;
  signal?: AbortSignal;
  timeoutMs?: number;
  responseType?: 'json' | 'blob';
  /**
   * Skip the global "session expired" handler on a 401. Used for login and for
   * session restoration, where a 401 is an expected answer rather than an expiry.
   */
  suppressUnauthorizedHandler?: boolean;
}

export interface UploadOptions {
  onProgress?: (percent: number) => void;
  signal?: AbortSignal;
  timeoutMs?: number;
  suppressUnauthorizedHandler?: boolean;
}

/* ------------------------------------------------------------------ */
/* Error parsing                                                       */
/* ------------------------------------------------------------------ */

interface ParsedErrorBody {
  message?: string;
  fieldErrors?: Record<string, string>;
}

/** Understands FastAPI-style `detail` payloads and common `{message}` / `{error}` shapes. */
export function parseErrorBody(body: unknown): ParsedErrorBody {
  if (typeof body === 'string') {
    const text = body.trim();
    return text !== '' && text.length < 300 && !text.startsWith('<') ? { message: text } : {};
  }
  if (!isRecord(body)) return {};

  const detail = body.detail ?? body.message ?? body.error;
  if (typeof detail === 'string') return { message: detail };
  if (isRecord(detail) && typeof detail.message === 'string') return { message: detail.message };

  if (Array.isArray(detail)) {
    const fieldErrors: Record<string, string> = {};
    const messages: string[] = [];
    for (const item of detail) {
      if (isRecord(item) && typeof item.msg === 'string') {
        messages.push(item.msg);
        const location = Array.isArray(item.loc)
          ? item.loc.filter((part): part is string => typeof part === 'string' && part !== 'body')
          : [];
        const field = location[location.length - 1];
        if (field && !(field in fieldErrors)) fieldErrors[field] = item.msg;
      }
    }
    return {
      message: messages[0],
      fieldErrors: Object.keys(fieldErrors).length > 0 ? fieldErrors : undefined,
    };
  }
  return {};
}

export function errorFromResponse(status: number, text: string): ApiError {
  let parsed: unknown = undefined;
  if (text !== '') {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  const { message, fieldErrors } = parseErrorBody(parsed);
  const kind = kindFromStatus(status);
  return new ApiError({ kind, status, message: message ?? defaultMessageFor(kind), fieldErrors });
}

function isAbortError(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'name' in error &&
    ((error as { name: unknown }).name === 'AbortError' ||
      (error as { name: unknown }).name === 'TimeoutError')
  );
}

function toTransportError(error: unknown, timedOut: boolean): ApiError {
  if (error instanceof ApiError) return error;
  if (isAbortError(error)) {
    const kind = timedOut ? 'timeout' : 'aborted';
    return new ApiError({ kind, message: defaultMessageFor(kind) });
  }
  return new ApiError({ kind: 'network', message: defaultMessageFor('network') });
}

/** Combines a caller's AbortSignal with a timeout. */
function linkSignals(external: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onAbort = () => controller.abort();

  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener('abort', onAbort, { once: true });
  }

  return {
    signal: controller.signal,
    didTimeout: () => timedOut,
    cleanup: () => {
      clearTimeout(timer);
      external?.removeEventListener('abort', onAbort);
    },
  };
}

/* ------------------------------------------------------------------ */
/* Client                                                              */
/* ------------------------------------------------------------------ */

export class ApiClient {
  private unauthorizedHandler: (() => void) | null = null;
  private readonly baseUrl: string;
  private readonly defaultTimeoutMs: number;
  private token: string | null =
    typeof window !== 'undefined' ? window.sessionStorage.getItem('medtrust_token') : null;

  constructor(baseUrl: string, defaultTimeoutMs: number) {
    this.baseUrl = baseUrl;
    this.defaultTimeoutMs = defaultTimeoutMs;
  }

  setToken(token: string | null): void {
    this.token = token;
    if (typeof window !== 'undefined') {
      if (token) window.sessionStorage.setItem('medtrust_token', token);
      else window.sessionStorage.removeItem('medtrust_token');
    }
  }

  getToken(): string | null {
    return this.token;
  }

  /** Called whenever an authenticated request is answered with 401. */
  setUnauthorizedHandler(handler: (() => void) | null): void {
    this.unauthorizedHandler = handler;
  }

  url(path: string, query?: Record<string, Primitive>): string {
    const url = `${this.baseUrl}${path}`;
    if (!query) return url;
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== '') params.set(key, String(value));
    }
    const qs = params.toString();
    return qs === '' ? url : `${url}?${qs}`;
  }

  /** Performs a JSON (or blob) request. Resolves with the parsed body, typed `unknown` by default. */
  async request<T = unknown>(path: string, options: RequestOptions = {}): Promise<T> {
    const {
      method = 'GET',
      body,
      query,
      signal,
      timeoutMs = this.defaultTimeoutMs,
      responseType = 'json',
      suppressUnauthorizedHandler = false,
    } = options;

    const link = linkSignals(signal, timeoutMs);
    const headers: Record<string, string> = {
      Accept: responseType === 'json' ? 'application/json' : '*/*',
      'Bypass-Tunnel-Reminder': 'true',
    };
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }
    let payload: BodyInit | undefined;
    if (body instanceof FormData) {
      payload = body;
    } else if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      payload = JSON.stringify(body);
    }

    let response: Response;
    try {
      response = await fetch(this.url(path, query), {
        method,
        headers,
        body: payload,
        credentials: 'include',
        signal: link.signal,
      });
    } catch (error) {
      link.cleanup();
      throw toTransportError(error, link.didTimeout());
    }

    try {
      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw errorFromResponse(response.status, text);
      }
      if (response.status === 204) return undefined as T;
      if (responseType === 'blob') return (await response.blob()) as T;

      const text = await response.text();
      if (text === '') return undefined as T;
      try {
        return JSON.parse(text) as T;
      } catch {
        throw new ApiError({
          kind: 'bad_response',
          status: response.status,
          message: defaultMessageFor('bad_response'),
        });
      }
    } catch (error) {
      const apiError = toTransportError(error, link.didTimeout());
      if (apiError.kind === 'unauthorized' && !suppressUnauthorizedHandler) {
        this.unauthorizedHandler?.();
      }
      throw apiError;
    } finally {
      link.cleanup();
    }
  }

  /**
   * Uploads multipart form data using XMLHttpRequest, because fetch cannot
   * report upload progress.
   */
  upload<T = unknown>(path: string, form: FormData, options: UploadOptions = {}): Promise<T> {
    const {
      onProgress,
      signal,
      timeoutMs = this.defaultTimeoutMs,
      suppressUnauthorizedHandler = false,
    } = options;

    return new Promise<T>((resolve, reject) => {
      if (signal?.aborted) {
        reject(new ApiError({ kind: 'aborted', message: defaultMessageFor('aborted') }));
        return;
      }

      const xhr = new XMLHttpRequest();
      let settled = false;
      const onAbort = () => xhr.abort();

      const finish = (action: () => void) => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener('abort', onAbort);
        action();
      };
      const fail = (error: ApiError) =>
        finish(() => {
          if (error.kind === 'unauthorized' && !suppressUnauthorizedHandler) {
            this.unauthorizedHandler?.();
          }
          reject(error);
        });

      signal?.addEventListener('abort', onAbort, { once: true });

      xhr.open('POST', this.url(path));
      xhr.withCredentials = true;
      xhr.timeout = timeoutMs;
      xhr.responseType = 'text';
      xhr.setRequestHeader('Accept', 'application/json');
      if (this.token) {
        xhr.setRequestHeader('Authorization', `Bearer ${this.token}`);
      }

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(Math.min(100, Math.round((event.loaded / event.total) * 100)));
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          let parsed: unknown;
          try {
            parsed = xhr.responseText === '' ? undefined : JSON.parse(xhr.responseText);
          } catch {
            fail(
              new ApiError({
                kind: 'bad_response',
                status: xhr.status,
                message: defaultMessageFor('bad_response'),
              }),
            );
            return;
          }
          finish(() => resolve(parsed as T));
        } else {
          fail(errorFromResponse(xhr.status, xhr.responseText));
        }
      };
      xhr.onerror = () =>
        fail(new ApiError({ kind: 'network', message: defaultMessageFor('network') }));
      xhr.ontimeout = () =>
        fail(new ApiError({ kind: 'timeout', message: defaultMessageFor('timeout') }));
      xhr.onabort = () =>
        fail(new ApiError({ kind: 'aborted', message: defaultMessageFor('aborted') }));

      xhr.send(form);
    });
  }
}

export const apiClient = new ApiClient(env.apiBaseUrl, env.requestTimeoutMs);

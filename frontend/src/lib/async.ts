import { ApiError, defaultMessageFor } from '@/services/errors';

/** Resolves after `ms`, or rejects with an "aborted" ApiError if the signal fires first. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new ApiError({ kind: 'aborted', message: defaultMessageFor('aborted') }));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new ApiError({ kind: 'aborted', message: defaultMessageFor('aborted') }));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

import { useEffect, useRef, useState } from 'react';

export interface AsyncState<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  /** Re-runs the loader. */
  reload: () => void;
}

interface Internal<T> {
  data?: T;
  error?: unknown;
  loading: boolean;
}

/**
 * Runs `loader` whenever `key` changes (or `reload()` is called), aborting any
 * in-flight request. The loader is read from a ref, so it can close over props
 * and state freely; encode everything it depends on into `key`.
 */
export function useAsyncData<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  key: string,
  enabled = true,
): AsyncState<T> {
  const loaderRef = useRef(loader);
  const [state, setState] = useState<Internal<T>>({ loading: enabled });
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    loaderRef.current = loader;
  });

  useEffect(() => {
    if (!enabled) {
      setState({ loading: false });
      return;
    }
    const controller = new AbortController();
    setState({ loading: true });
    loaderRef
      .current(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ data, loading: false });
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setState({ error, loading: false });
      });
    return () => controller.abort();
  }, [key, enabled, reloadToken]);

  return {
    data: state.data,
    error: state.error,
    loading: state.loading,
    reload: () => setReloadToken((token) => token + 1),
  };
}

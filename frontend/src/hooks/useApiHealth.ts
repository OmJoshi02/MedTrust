import { env } from '@/config/env';
import { analysisService, describeError } from '@/services';
import type { HealthStatus } from '@/types/api';
import { useAsyncData } from './useAsyncData';

export type ApiHealthState = 'demo' | 'checking' | 'ready' | 'not_ready' | 'offline';

export interface ApiHealth {
  state: ApiHealthState;
  detail?: string;
  recheck: () => void;
}

interface HealthResult {
  live: HealthStatus;
  ready: HealthStatus;
}

/** Checks backend liveness and readiness once on mount, and on demand. */
export function useApiHealth(): ApiHealth {
  const { data, error, loading, reload } = useAsyncData<HealthResult>(
    async (signal) => {
      const live = await analysisService.healthLive(signal);
      const ready = await analysisService.healthReady(signal);
      return { live, ready };
    },
    'api-health',
    !env.demoMode,
  );

  if (env.demoMode) return { state: 'demo', recheck: reload };
  if (loading) return { state: 'checking', recheck: reload };
  if (error || !data) return { state: 'offline', detail: describeError(error), recheck: reload };
  if (!data.ready.ok) return { state: 'not_ready', detail: data.ready.detail, recheck: reload };
  return { state: 'ready', recheck: reload };
}

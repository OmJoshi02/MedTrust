import { env } from '@/config/env';

/** Persistent strip shown on every page while demo mode is enabled. */
export function DemoBanner() {
  if (!env.demoMode) return null;
  return (
    <div role="note" className="no-print border-b border-ink/10 bg-ink px-4 py-2 text-center text-xs text-surface/90">
      <strong className="font-semibold text-surface">Demo mode.</strong> All results are sample data, not model output,
      and sign-in is a non-secure local simulation. No backend is contacted.
    </div>
  );
}

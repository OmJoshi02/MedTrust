import { useApiHealth, type ApiHealthState } from '@/hooks/useApiHealth';
import { cn } from '@/lib/cn';

const COPY: Record<ApiHealthState, { label: string; dot: string }> = {
  demo: { label: 'Demo mode', dot: 'bg-muted' },
  checking: { label: 'Checking service', dot: 'animate-pulse bg-muted' },
  ready: { label: 'Service ready', dot: 'bg-success' },
  not_ready: { label: 'Service starting', dot: 'bg-warning' },
  offline: { label: 'Service unreachable', dot: 'bg-danger' },
};

/** Small development-friendly indicator of backend liveness and readiness. */
export function ApiStatusIndicator({ className }: { className?: string }) {
  const { state, detail, recheck } = useApiHealth();
  const copy = COPY[state];
  const interactive = state === 'offline' || state === 'not_ready';

  const content = (
    <>
      <span aria-hidden="true" className={cn('h-1.5 w-1.5 rounded-full', copy.dot)} />
      <span>{copy.label}</span>
    </>
  );

  const classes = cn('inline-flex items-center gap-2 text-xs text-muted', className);

  if (!interactive) {
    return (
      <span role="status" className={classes}>
        {content}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={recheck}
      title={detail ? `${detail} Click to check again.` : 'Click to check again.'}
      className={cn(classes, 'rounded hover:text-ink')}
    >
      {content}
    </button>
  );
}

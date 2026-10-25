import { CloudOff, LockKeyhole, RefreshCw, TriangleAlert } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { describeError, isApiError } from '@/services';
import { EmptyState } from './EmptyState';

interface ErrorStateProps {
  error: unknown;
  onRetry?: () => void;
  title?: string;
  className?: string;
}

/** Renders a thrown value as a clear state with a retry action. */
export function ErrorState({ error, onRetry, title, className }: ErrorStateProps) {
  const kind = isApiError(error) ? error.kind : 'unknown';
  const unreachable = kind === 'network' || kind === 'timeout';
  const denied = kind === 'unauthorized' || kind === 'forbidden';

  const heading =
    title ??
    (unreachable ? 'The service is unavailable' : denied ? 'Access denied' : 'Something went wrong');

  const icon = unreachable ? (
    <CloudOff aria-hidden="true" className="h-5 w-5" />
  ) : denied ? (
    <LockKeyhole aria-hidden="true" className="h-5 w-5" />
  ) : (
    <TriangleAlert aria-hidden="true" className="h-5 w-5" />
  );

  return (
    <div role="alert" className={className}>
      <EmptyState
        icon={icon}
        title={heading}
        description={describeError(error)}
        action={
          onRetry && (
            <Button variant="secondary" onClick={onRetry} leadingIcon={<RefreshCw aria-hidden="true" className="h-4 w-4" />}>
              Try again
            </Button>
          )
        }
      />
    </div>
  );
}

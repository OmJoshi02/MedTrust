import { cn } from '@/lib/cn';

/** Decorative loading placeholder. Pair with a visible or sr-only status message. */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={cn('skeleton animate-shimmer rounded-md', className)} />;
}

import { cn } from '@/lib/cn';

interface ProgressBarProps {
  /** 0–100. Omit for an indeterminate bar. */
  value?: number;
  label: string;
  className?: string;
}

export function ProgressBar({ value, label, className }: ProgressBarProps) {
  const determinate = value !== undefined;
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={determinate ? Math.round(value) : undefined}
      className={cn('relative h-1.5 w-full overflow-hidden rounded-full bg-subtle', className)}
    >
      {determinate ? (
        <div
          className="h-full rounded-full bg-ink transition-[width] duration-300 ease-out"
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      ) : (
        <div className="absolute inset-y-0 left-0 w-2/5 animate-indeterminate rounded-full bg-ink" />
      )}
    </div>
  );
}

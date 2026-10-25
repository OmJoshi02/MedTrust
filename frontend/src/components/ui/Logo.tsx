import { cn } from '@/lib/cn';

interface LogoProps {
  className?: string;
  /** "light" renders for use on dark backgrounds. */
  tone?: 'dark' | 'light';
  showMark?: boolean;
}

/** The MedTrust wordmark with its small monogram. */
export function Logo({ className, tone = 'dark', showMark = true }: LogoProps) {
  const light = tone === 'light';
  return (
    <span className={cn('inline-flex items-center gap-2.5', light ? 'text-surface' : 'text-ink', className)}>
      {showMark && (
        <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true" focusable="false">
          <rect width="32" height="32" rx="7" fill={light ? '#F7F7F5' : '#171717'} />
          <path
            d="M8.5 22.5V10l7.5 8.5L23.5 10v12.5"
            fill="none"
            stroke={light ? '#171717' : '#F7F7F5'}
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )}
      <span className="text-[1.0625rem] font-semibold tracking-[-0.02em]">MedTrust</span>
    </span>
  );
}

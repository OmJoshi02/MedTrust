import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

interface SectionShellProps {
  id: string;
  number?: string;
  title: string;
  description?: ReactNode;
  badge?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** Consistent heading + body wrapper for each report section. */
export function SectionShell({ id, number, title, description, badge, children, className }: SectionShellProps) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-title`}
      className={cn('print-avoid-break scroll-mt-28 border-t border-line pt-8', className)}
    >
      <header className="mb-6 flex flex-wrap items-start justify-between gap-x-6 gap-y-2">
        <div className="min-w-0">
          {number && <p className="eyebrow mb-2 tabular">{number}</p>}
          <h2 id={`${id}-title`} className="text-xl font-semibold tracking-tight sm:text-2xl">
            {title}
          </h2>
          {description && <p className="mt-2 max-w-prose text-sm leading-relaxed text-muted">{description}</p>}
        </div>
        {badge}
      </header>
      {children}
    </section>
  );
}

/** Explicit "not available" message so a missing section is never silently omitted. */
export function Unavailable({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-line bg-subtle/40 px-4 py-3.5 text-sm text-muted">
      {children}
    </div>
  );
}

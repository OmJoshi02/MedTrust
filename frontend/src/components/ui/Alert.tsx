import type { ReactNode } from 'react';
import { CircleAlert, CircleCheck, Info, TriangleAlert } from 'lucide-react';
import { cn } from '@/lib/cn';

export type AlertTone = 'info' | 'success' | 'warning' | 'danger';

const styles: Record<AlertTone, { box: string; icon: ReactNode }> = {
  info: {
    box: 'border-line bg-subtle/70 text-charcoal',
    icon: <Info aria-hidden="true" className="h-4 w-4 text-muted" />,
  },
  success: {
    box: 'border-success/25 bg-success/[0.06] text-ink',
    icon: <CircleCheck aria-hidden="true" className="h-4 w-4 text-success" />,
  },
  warning: {
    box: 'border-warning/30 bg-warning/[0.07] text-ink',
    icon: <TriangleAlert aria-hidden="true" className="h-4 w-4 text-warning" />,
  },
  danger: {
    box: 'border-danger/30 bg-danger/[0.06] text-ink',
    icon: <CircleAlert aria-hidden="true" className="h-4 w-4 text-danger" />,
  },
};

interface AlertProps {
  tone?: AlertTone;
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function Alert({ tone = 'info', title, children, action, className }: AlertProps) {
  const role = tone === 'danger' || tone === 'warning' ? 'alert' : 'status';
  return (
    <div role={role} className={cn('flex gap-3 rounded-md border px-4 py-3 text-sm', styles[tone].box, className)}>
      <span className="mt-0.5 shrink-0">{styles[tone].icon}</span>
      <div className="min-w-0 flex-1">
        {title && <p className="font-medium text-ink">{title}</p>}
        {children && <div className={cn('leading-relaxed', title && 'mt-0.5 text-charcoal')}>{children}</div>}
        {action && <div className="mt-2.5">{action}</div>}
      </div>
    </div>
  );
}

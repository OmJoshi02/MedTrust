import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

/** Centres content at the site width with a responsive gutter. */
export function PageContainer({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('page-container', className)} {...rest} />;
}

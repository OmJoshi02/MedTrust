import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

type PanelVariant = 'solid' | 'glass' | 'subtle' | 'dark';

const variants: Record<PanelVariant, string> = {
  solid: 'border border-line bg-surface',
  glass: 'glass',
  subtle: 'border border-line bg-subtle/60',
  dark: 'on-dark bg-dark text-surface',
};

interface PanelProps extends HTMLAttributes<HTMLElement> {
  as?: 'section' | 'div' | 'article' | 'aside';
  variant?: PanelVariant;
  padded?: boolean;
}

/** A restrained bordered surface. Use sparingly: whitespace and hairlines come first. */
export function Panel({ as: Tag = 'section', variant = 'solid', padded = true, className, ...rest }: PanelProps) {
  return <Tag className={cn('rounded-lg', variants[variant], padded && 'p-5 sm:p-6', className)} {...rest} />;
}

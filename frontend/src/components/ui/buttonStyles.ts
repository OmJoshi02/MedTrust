import { cn } from '@/lib/cn';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'inverse' | 'outline-inverse' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

const base =
  'inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium transition-colors duration-150 disabled:pointer-events-none';

const variants: Record<ButtonVariant, string> = {
  primary: 'bg-ink text-surface hover:bg-charcoal active:bg-black disabled:bg-ink/25 disabled:text-surface',
  secondary:
    'border border-line bg-surface text-ink hover:border-ink/40 hover:bg-subtle/60 disabled:text-muted disabled:opacity-60',
  ghost: 'text-ink hover:bg-subtle disabled:text-muted disabled:opacity-60',
  inverse: 'bg-surface text-ink hover:bg-subtle disabled:opacity-50',
  'outline-inverse': 'border border-white/25 text-surface hover:bg-white/10 disabled:opacity-50',
  danger: 'bg-red-600 text-white hover:bg-red-700 active:bg-red-800 disabled:bg-red-300 disabled:opacity-60',
};

const sizes: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-[0.8125rem]',
  md: 'h-10 px-4 text-sm',
  lg: 'h-12 px-6 text-[0.9375rem]',
};

interface StyleOptions {
  variant?: ButtonVariant;
  size?: ButtonSize;
  className?: string;
}

export function buttonStyles({ variant = 'primary', size = 'md', className }: StyleOptions = {}): string {
  return cn(base, variants[variant], sizes[size], className);
}

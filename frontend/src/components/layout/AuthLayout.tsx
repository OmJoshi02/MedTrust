import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Logo } from '@/components/ui/Logo';

interface AuthLayoutProps {
  title: string;
  subtitle: ReactNode;
  children: ReactNode;
  footer: ReactNode;
}

const POINTS = [
  'Calibrated probabilities for 14 chest findings',
  'Image-quality and out-of-distribution checks',
  'Uncertainty and explainability alongside every result',
];

/** Split-screen frame shared by login and sign-up: editorial panel + glass form panel. */
export function AuthLayout({ title, subtitle, children, footer }: AuthLayoutProps) {
  return (
    <div className="grid min-h-[calc(100vh-4rem)] lg:grid-cols-[1fr_1.05fr]">
      <aside className="on-dark relative hidden overflow-hidden bg-dark text-surface lg:flex lg:flex-col lg:justify-between lg:p-14">
        <div aria-hidden="true" className="bg-grid absolute inset-0 opacity-40 invert" />
        <Link to="/" aria-label="MedTrust home" className="relative w-fit rounded">
          <Logo tone="light" />
        </Link>
        <div className="relative">
          <p className="eyebrow !text-surface/50">Research application</p>
          <p className="mt-4 max-w-md text-3xl font-semibold leading-tight tracking-[-0.03em]">
            Understand the prediction. Evaluate its reliability.
          </p>
          <ul className="mt-8 space-y-3 text-sm text-surface/70">
            {POINTS.map((point) => (
              <li key={point} className="flex gap-3">
                <span aria-hidden="true" className="mt-2 h-px w-4 shrink-0 bg-surface/50" />
                {point}
              </li>
            ))}
          </ul>
        </div>
        <p className="relative max-w-sm text-xs leading-relaxed text-surface/45">
          Research use only. Not a medical device and not intended for diagnosis or clinical decisions.
        </p>
      </aside>

      <div className="flex items-center justify-center px-[var(--gutter)] py-12 sm:py-16">
        <div className="glass w-full max-w-md rounded-xl p-6 sm:p-9">
          <h1 className="text-[1.75rem] font-semibold leading-tight tracking-[-0.03em]">{title}</h1>
          <p className="mt-2 text-sm leading-relaxed text-muted">{subtitle}</p>
          <div className="mt-7">{children}</div>
          <div className="mt-7 border-t border-line pt-5 text-sm text-muted">{footer}</div>
        </div>
      </div>
    </div>
  );
}

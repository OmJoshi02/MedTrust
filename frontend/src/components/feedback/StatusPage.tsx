import type { ReactNode } from 'react';
import { PageContainer } from '@/components/layout/PageContainer';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';

interface StatusPageProps {
  code?: string;
  title: string;
  description: ReactNode;
  actions: ReactNode;
  icon?: ReactNode;
  documentTitle: string;
}

/** Full-page state used for 404, unauthorized, API-unavailable and runtime errors. */
export function StatusPage({ code, title, description, actions, icon, documentTitle }: StatusPageProps) {
  useDocumentTitle(documentTitle);
  return (
    <PageContainer className="flex min-h-[60vh] flex-col items-start justify-center py-20 animate-page-in">
      {icon ? (
        <span className="mb-6 inline-flex h-12 w-12 items-center justify-center rounded-full border border-line bg-surface text-muted">
          {icon}
        </span>
      ) : (
        code && <p className="tabular mb-6 text-sm font-medium tracking-[0.2em] text-muted">{code}</p>
      )}
      <h1 className="h-page max-w-2xl">{title}</h1>
      <p className="lead mt-4 max-w-prose">{description}</p>
      <div className="mt-8 flex flex-wrap gap-3">{actions}</div>
    </PageContainer>
  );
}

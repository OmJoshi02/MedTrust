import { useEffect } from 'react';

const SUFFIX = 'MedTrust';

export function useDocumentTitle(title: string | null): void {
  useEffect(() => {
    document.title = title ? `${title} — ${SUFFIX}` : `${SUFFIX} — Trust-Aware Explainable AI for Chest X-Ray Analysis`;
  }, [title]);
}

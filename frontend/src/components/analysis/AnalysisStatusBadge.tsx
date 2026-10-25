import { Badge, type BadgeTone } from '@/components/ui/Badge';
import type { AnalysisStatus } from '@/types/analysis';

const COPY: Record<AnalysisStatus, { label: string; tone: BadgeTone }> = {
  completed: { label: 'Completed', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
  processing: { label: 'Processing', tone: 'warning' },
  queued: { label: 'Queued', tone: 'neutral' },
  unknown: { label: 'Unknown', tone: 'neutral' },
};

export function AnalysisStatusBadge({ status }: { status: AnalysisStatus }) {
  const copy = COPY[status];
  return <Badge tone={copy.tone}>{copy.label}</Badge>;
}

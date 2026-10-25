import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { formatMetricValue } from './metricFormat';
import type { MetricStatus, ReportMetric } from '@/types/analysis';

const STATUS_COPY: Record<MetricStatus, { label: string; tone: BadgeTone }> = {
  good: { label: 'Adequate', tone: 'success' },
  caution: { label: 'Borderline', tone: 'warning' },
  poor: { label: 'Poor', tone: 'danger' },
  info: { label: 'Info', tone: 'neutral' },
};

export function StatusBadge({ status }: { status: MetricStatus | undefined }) {
  if (!status || status === 'info') return null;
  const copy = STATUS_COPY[status];
  return <Badge tone={copy.tone}>{copy.label}</Badge>;
}

interface MetricListProps {
  metrics: ReportMetric[];
  /** Fallback explanation lookup used when the backend supplies no description. */
  explain: (key: string) => string | undefined;
}

/** Definition list of metrics with value, status and a short explanation each. */
export function MetricList({ metrics, explain }: MetricListProps) {
  return (
    <dl className="divide-y divide-line border-y border-line">
      {metrics.map((metric) => {
        const description = metric.description ?? explain(metric.key);
        return (
          <div key={metric.key} className="grid gap-x-8 gap-y-1 py-4 sm:grid-cols-[minmax(0,14rem)_1fr]">
            <dt className="flex items-center gap-2 text-sm font-medium">
              {metric.label}
              <StatusBadge status={metric.status} />
            </dt>
            <dd className="min-w-0">
              <p className="tabular text-base font-semibold tracking-tight">{formatMetricValue(metric)}</p>
              {description ? (
                <p className="mt-1 text-[0.8125rem] leading-relaxed text-muted">{description}</p>
              ) : null}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}

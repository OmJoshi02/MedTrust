import { qualityMetricExplanation } from '@/lib/explanations';
import type { QualityAssessment as QualityData } from '@/types/analysis';
import { MetricList, StatusBadge } from './MetricList';
import { Unavailable } from './SectionShell';

export function QualityAssessment({ quality }: { quality: QualityData | undefined }) {
  if (!quality) {
    return <Unavailable>Image-quality assessment was not returned for this analysis. It was not evaluated.</Unavailable>;
  }
  return (
    <div className="space-y-5">
      {(quality.summary || quality.overallStatus) && (
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={quality.overallStatus} />
          {quality.summary && <p className="max-w-prose text-sm leading-relaxed text-charcoal">{quality.summary}</p>}
        </div>
      )}
      {quality.metrics.length > 0 ? (
        <MetricList metrics={quality.metrics} explain={qualityMetricExplanation} />
      ) : (
        <Unavailable>No individual quality metrics were returned.</Unavailable>
      )}
    </div>
  );
}

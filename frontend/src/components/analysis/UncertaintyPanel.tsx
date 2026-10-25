import { uncertaintyMetricExplanation } from '@/lib/explanations';
import { formatNumber } from '@/lib/format';
import { DISEASES, formatDiseaseName, type UncertaintyAssessment } from '@/types/analysis';
import { MetricList } from './MetricList';
import { Unavailable } from './SectionShell';

export function UncertaintyPanel({ uncertainty }: { uncertainty: UncertaintyAssessment | undefined }) {
  if (!uncertainty) {
    return <Unavailable>Uncertainty estimates were not returned for this analysis. They were not evaluated.</Unavailable>;
  }

  const perDisease = DISEASES.filter((d) => uncertainty.perDisease[d] !== undefined);
  const maxValue = Math.max(0.0001, ...perDisease.map((d) => uncertainty.perDisease[d] ?? 0));

  return (
    <div className="space-y-8">
      {(uncertainty.method || uncertainty.summary) && (
        <div className="space-y-1.5">
          {uncertainty.method && (
            <p className="text-sm">
              <span className="text-muted">Method: </span>
              <span className="font-medium">{uncertainty.method}</span>
            </p>
          )}
          {uncertainty.summary && <p className="max-w-prose text-sm leading-relaxed text-charcoal">{uncertainty.summary}</p>}
        </div>
      )}

      {uncertainty.metrics.length > 0 && <MetricList metrics={uncertainty.metrics} explain={uncertaintyMetricExplanation} />}

      {perDisease.length > 0 && (
        <div>
          <h3 className="mb-1 text-sm font-semibold">Uncertainty by finding</h3>
          <p className="mb-4 text-[0.8125rem] text-muted">
            Bars are scaled to the largest value in this report. Higher means the output for that finding is less settled.
          </p>
          <ul className="divide-y divide-line/70 border-y border-line">
            {perDisease.map((disease) => {
              const value = uncertainty.perDisease[disease] ?? 0;
              return (
                <li key={disease} className="grid grid-cols-[minmax(0,9rem)_1fr_5rem] items-center gap-3 py-2.5 text-sm sm:grid-cols-[11rem_1fr_5.5rem]">
                  <span className="truncate">{formatDiseaseName(disease)}</span>
                  <div className="h-1.5 rounded-full bg-subtle" aria-hidden="true">
                    <div className="h-full rounded-full bg-charcoal" style={{ width: `${(value / maxValue) * 100}%` }} />
                  </div>
                  <span className="tabular text-right text-muted">{formatNumber(value, 5)}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {uncertainty.metrics.length === 0 && perDisease.length === 0 && (
        <Unavailable>No uncertainty values were returned.</Unavailable>
      )}

      <p className="max-w-prose text-[0.8125rem] leading-relaxed text-muted">
        An uncertainty value describes how settled the model&rsquo;s output is. It is not a measure of whether the
        prediction is correct, and a low value does not confirm a finding.
      </p>
    </div>
  );
}

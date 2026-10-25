import { ArrowRight, CircleAlert } from 'lucide-react';
import { DemoBadge } from '@/components/feedback/DemoBadge';
import { Alert } from '@/components/ui/Alert';
import { Badge } from '@/components/ui/Badge';
import { ButtonLink } from '@/components/ui/Button';
import { Panel } from '@/components/ui/Panel';
import { qualityMetricExplanation } from '@/lib/explanations';
import { formatDateTime, formatNumber } from '@/lib/format';
import { hasPredictions, type AnalysisReport } from '@/types/analysis';
import { StatusBadge } from './MetricList';
import { formatMetricValue } from './metricFormat';

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="border-t border-line py-3.5">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-1 text-sm font-medium">{children}</dd>
    </div>
  );
}

const NOT_EVALUATED = <span className="font-normal text-muted">Not evaluated</span>;

/** Concise preview of an analysis, with a clear route into the full report. */
export function ResultsPreview({ report }: { report: AnalysisReport }) {
  const failed = report.status === 'failed';
  const oodLabel =
    report.ood?.status === 'in_distribution'
      ? 'In distribution'
      : report.ood?.status === 'out_of_distribution'
        ? 'Out of distribution'
        : report.ood?.status === 'uncertain'
          ? 'Uncertain'
          : undefined;

  return (
    <Panel variant="glass" className="animate-page-in">
      <header className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="eyebrow mb-1.5">Analysis complete</p>
          <h2 className="truncate text-xl font-semibold tracking-tight" title={report.filename}>
            {report.filename ?? 'Uploaded image'}
          </h2>
          <p className="mt-1 text-xs text-muted">
            ID <span className="tabular font-mono">{report.id}</span> &middot; {formatDateTime(report.createdAt)}
          </p>
        </div>
        {report.isDemo && <DemoBadge label="Sample results — not model output" />}
      </header>

      {failed && (
        <Alert tone="danger" title="The analysis did not complete.">
          {report.error ?? 'The backend reported a failure without further detail.'}
        </Alert>
      )}

      {report.warnings.length > 0 && (
        <Alert tone="warning" title="Notes about this result" className="mb-5">
          <ul className="list-disc pl-4">
            {report.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </Alert>
      )}

      {!failed && (
        <>
          {report.quality ? (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pb-3">
                <div className="flex items-center gap-2.5">
                  <h3 className="text-base font-semibold text-ink">Image Quality Assessment</h3>
                  <StatusBadge status={report.quality.overallStatus} />
                </div>
                {report.quality.metrics.find((m) => m.key === 'composite_score') && (
                  <Badge
                    tone={
                      report.quality.overallStatus === 'good'
                        ? 'success'
                        : report.quality.overallStatus === 'poor'
                        ? 'danger'
                        : 'warning'
                    }
                  >
                    Quality Score: {report.quality.metrics.find((m) => m.key === 'composite_score')?.value}/100
                  </Badge>
                )}
              </div>

              {report.quality.summary && (
                <p className="rounded-md border border-line/60 bg-subtle/70 px-3.5 py-2.5 text-xs leading-relaxed text-charcoal">
                  {report.quality.summary}
                </p>
              )}

              <div className="overflow-x-auto">
                <table className="w-full min-w-[22rem] border-collapse text-sm">
                  <caption className="sr-only">Technical radiograph quality indicators</caption>
                  <thead>
                    <tr className="border-b border-line text-left text-xs text-muted">
                      <th scope="col" className="py-2.5 pr-4 font-medium">Indicator</th>
                      <th scope="col" className="py-2.5 pr-4 font-medium text-right">Measured Value</th>
                      <th scope="col" className="py-2.5 pr-4 font-medium text-center">Status</th>
                      <th scope="col" className="py-2.5 font-medium">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.quality.metrics.map((metric) => {
                      const description = metric.description ?? qualityMetricExplanation(metric.key);
                      return (
                        <tr key={metric.key} className="border-b border-line/70 last:border-0 hover:bg-subtle/30 transition-colors">
                          <th scope="row" className="py-3 pr-4 text-left font-medium text-ink">
                            {metric.label}
                          </th>
                          <td className="py-3 pr-4 text-right font-semibold tabular text-ink">
                            {formatMetricValue(metric)}
                          </td>
                          <td className="py-3 pr-4 text-center">
                            <StatusBadge status={metric.status} />
                          </td>
                          <td className="py-3 text-xs leading-relaxed text-muted">
                            {description ?? 'Technical radiograph indicator.'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <Alert tone="info" title="No image-quality assessment was returned.">
              The backend response did not include image-quality evaluation for this radiograph.
            </Alert>
          )}

          <dl className="mt-6 grid gap-x-8 sm:grid-cols-2">
            <Fact label="Disease predictions">
              {hasPredictions(report) ? (
                <span className="text-sm font-medium text-ink">
                  {report.predictions.length} pathologies evaluated · Available in full report
                </span>
              ) : (
                NOT_EVALUATED
              )}
            </Fact>
            <Fact label="Out-of-distribution">
              {report.ood ? (
                <span className="flex flex-wrap items-center gap-2">
                  {oodLabel ?? 'Status not provided'}
                  {report.ood.score !== undefined && (
                    <Badge tone="neutral">score {formatNumber(report.ood.score, 2)}</Badge>
                  )}
                </span>
              ) : (
                NOT_EVALUATED
              )}
            </Fact>
            <Fact label="Uncertainty">{report.uncertainty ? 'MC-Dropout (30 passes) · Available in report' : NOT_EVALUATED}</Fact>
            <Fact label="Calibration">{report.calibration ? 'Temperature scaled · Available in report' : NOT_EVALUATED}</Fact>
            <Fact label="Explainability">
              {report.explanation && report.explanation.artifacts.length > 0
                ? `${report.explanation.artifacts.length} Grad-CAM heatmaps · Available in report`
                : NOT_EVALUATED}
            </Fact>
          </dl>
        </>
      )}

      <p className="mt-5 flex items-start gap-2 text-xs leading-relaxed text-muted">
        <CircleAlert aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        Image quality results are engineering indicators for technical evaluation. Complete 14-disease predictions, Grad-CAM overlays, and uncertainty analysis are accessible in the full report.
      </p>

      <div className="mt-6 flex flex-wrap gap-3">
        <ButtonLink
          to={`/report/${encodeURIComponent(report.id)}`}
          size="lg"
          trailingIcon={<ArrowRight aria-hidden="true" className="h-4 w-4" />}
        >
          Open full report
        </ButtonLink>
      </div>
    </Panel>
  );
}

import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { formatNumber } from '@/lib/format';
import type { OODAssessment, OODStatus } from '@/types/analysis';
import { Unavailable } from './SectionShell';

const STATUS: Record<OODStatus, { label: string; tone: BadgeTone }> = {
  in_distribution: { label: 'In distribution', tone: 'success' },
  out_of_distribution: { label: 'Out of distribution', tone: 'warning' },
  uncertain: { label: 'Uncertain', tone: 'warning' },
};

export function OODPanel({ ood }: { ood: OODAssessment | undefined }) {
  if (!ood) {
    return <Unavailable>Out-of-distribution assessment was not returned for this analysis. It was not evaluated.</Unavailable>;
  }
  const status = ood.status ? STATUS[ood.status] : undefined;

  return (
    <div className="space-y-5">
      <dl className="grid gap-x-8 gap-y-5 sm:grid-cols-3">
        <div>
          <dt className="text-xs text-muted">Status</dt>
          <dd className="mt-1.5 flex flex-wrap items-center gap-2">
            {status ? <Badge tone={status.tone}>{status.label}</Badge> : <span className="text-sm text-muted">Not provided</span>}
            {ood.triggersCount !== undefined && ood.triggersTotal !== undefined && (
              <span className="text-xs text-muted">
                ({ood.triggersCount} of {ood.triggersTotal} triggered)
              </span>
            )}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-muted">
            Lead Metric ({ood.leadDetectorName ?? 'Mahalanobis Distance'})
          </dt>
          <dd className="tabular mt-1 text-xl font-semibold tracking-tight">
            {ood.score === undefined || ood.score === null ? (
              <span className="text-sm font-normal text-muted">Not available</span>
            ) : (
              formatNumber(ood.score, 3)
            )}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-muted">Lead Metric Threshold</dt>
          <dd className="tabular mt-1 text-xl font-semibold tracking-tight">
            {ood.threshold === undefined || ood.threshold === null ? (
              <span className="text-sm font-normal text-muted">Not available</span>
            ) : (
              formatNumber(ood.threshold, 3)
            )}
          </dd>
        </div>
      </dl>

      {ood.method && (
        <p className="text-sm">
          <span className="text-muted">Method: </span>
          <span className="font-medium">{ood.method}</span>
        </p>
      )}
      {ood.explanation && <p className="max-w-prose text-sm leading-relaxed text-charcoal">{ood.explanation}</p>}

      {ood.detectors && ood.detectors.length > 0 && (
        <div className="overflow-hidden rounded-md border border-line bg-surface">
          <div className="border-b border-line bg-subtle/40 px-3.5 py-2 text-xs font-medium text-muted">
            Individual Detector Statistics (Validation Calibrated)
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-line bg-subtle/20 text-muted">
                <tr>
                  <th className="py-2 px-3.5 font-medium">Detector</th>
                  <th className="py-2 px-3.5 font-medium">Observed Score</th>
                  <th className="py-2 px-3.5 font-medium">Decision Threshold</th>
                  <th className="py-2 px-3.5 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line font-mono text-[0.8125rem]">
                {ood.detectors.map((d) => (
                  <tr key={d.key || d.name} className="hover:bg-subtle/20">
                    <td className="py-2 px-3.5 font-sans font-medium text-charcoal">
                      <div>{d.name}</div>
                      {d.description && <div className="text-[0.7rem] font-normal text-muted">{d.description}</div>}
                    </td>
                    <td className="py-2 px-3.5">
                      {d.score === undefined || d.score === null ? 'Not available' : formatNumber(d.score, 3)}
                    </td>
                    <td className="py-2 px-3.5 text-muted">
                      {d.threshold === undefined || d.threshold === null ? 'Not available' : formatNumber(d.threshold, 3)}
                    </td>
                    <td className="py-2 px-3.5 font-sans">
                      <Badge tone={d.is_ood ? 'warning' : 'success'}>
                        {d.is_ood ? 'Out of distribution' : 'In distribution'}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <p className="max-w-prose rounded-md border border-line bg-subtle/50 px-4 py-3 text-[0.8125rem] leading-relaxed text-muted">
        The OOD score measures how unlike the model&rsquo;s development data this image appears. It is{' '}
        <strong className="font-medium text-charcoal">not a probability of disease</strong>. An image flagged as out of
        distribution is one for which the predictions above are less trustworthy.
      </p>
    </div>
  );
}

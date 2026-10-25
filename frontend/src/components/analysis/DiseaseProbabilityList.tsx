import { useState } from 'react';
import { Segmented } from '@/components/ui/Segmented';
import { cn } from '@/lib/cn';
import { formatPercent } from '@/lib/format';
import { formatDiseaseName, type DiseasePrediction } from '@/types/analysis';

type View = 'calibrated' | 'raw' | 'both';

interface DiseaseProbabilityListProps {
  predictions: DiseasePrediction[];
  /** "compact" is for result previews: a single view, no thresholds. */
  variant?: 'full' | 'compact';
}

function Bar({ value, tone, threshold }: { value: number | null; tone: 'solid' | 'outline'; threshold?: number | null }) {
  if (value === null) {
    return <div className="h-1.5 rounded-full border border-dashed border-line" aria-hidden="true" />;
  }
  return (
    <div className="relative h-1.5 rounded-full bg-subtle" aria-hidden="true">
      <div
        className={cn('h-full rounded-full', tone === 'solid' ? 'bg-ink' : 'bg-ink/35')}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
      />
      {threshold !== null && threshold !== undefined && (
        <span
          className="absolute -top-0.5 h-2.5 w-px bg-charcoal"
          style={{ left: `${Math.max(0, Math.min(1, threshold)) * 100}%` }}
        />
      )}
    </div>
  );
}

/**
 * The 14 findings in canonical order. A predicted probability is the model's
 * estimate, not a confirmed diagnosis. Values missing from the response are
 * shown as "Not provided", never filled in.
 */
export function DiseaseProbabilityList({ predictions, variant = 'full' }: DiseaseProbabilityListProps) {
  const hasCalibrated = predictions.some((p) => p.calibratedProbability !== null);
  const hasRaw = predictions.some((p) => p.rawProbability !== null);
  const hasThresholds = predictions.some((p) => p.threshold !== null);
  const compact = variant === 'compact';

  const [chosen, setChosen] = useState<View>('both');
  const [sortBy, setSortBy] = useState<'highest' | 'canonical'>('canonical');

  const available: View = hasCalibrated && hasRaw ? chosen : hasCalibrated ? 'calibrated' : 'raw';
  const view: View = compact && available === 'both' ? 'calibrated' : available;

  const showCalibrated = view === 'calibrated' || view === 'both';
  const showRaw = view === 'raw' || view === 'both';

  // Arrange findings from highest prediction to lowest by default
  const sortedPredictions = [...predictions].sort((a, b) => {
    if (sortBy === 'canonical') return 0;
    const scoreA =
      view === 'raw'
        ? (a.rawProbability ?? -1)
        : (a.calibratedProbability ?? a.rawProbability ?? -1);
    const scoreB =
      view === 'raw'
        ? (b.rawProbability ?? -1)
        : (b.calibratedProbability ?? b.rawProbability ?? -1);
    return scoreB - scoreA;
  });

  return (
    <div>
      {!compact && (
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            {hasCalibrated && hasRaw && (
              <Segmented<View>
                ariaLabel="Probability view"
                value={chosen}
                onChange={setChosen}
                options={[
                  { value: 'both', label: 'Both' },
                  { value: 'calibrated', label: 'Calibrated' },
                  { value: 'raw', label: 'Raw' },
                ]}
              />
            )}
            <Segmented<'highest' | 'canonical'>
              ariaLabel="Sort order"
              value={sortBy}
              onChange={setSortBy}
              options={[
                { value: 'canonical', label: 'Standard Order' },
                { value: 'highest', label: 'Highest to Lowest' },
              ]}
            />
          </div>
          {hasThresholds && (
            <p className="text-xs text-muted">
              Tick mark = decision threshold ({view === 'raw' ? 'Youden-J raw space' : view === 'both' ? 'calibrated & raw equivalents' : 'calibrated equivalent'})
            </p>
          )}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[22rem] border-collapse text-sm">
          <caption className="sr-only">
            Predicted probability for each of the 14 findings, arranged from highest to lowest prediction. These are model estimates, not confirmed diagnoses.
          </caption>
          <thead>
            <tr className="border-b border-line text-left text-xs text-muted">
              <th scope="col" className="w-10 py-2 pr-2 font-medium tabular">
                #
              </th>
              <th scope="col" className="py-2 pr-4 font-medium">
                Finding
              </th>
              <th scope="col" className="hidden py-2 pr-4 font-medium sm:table-cell">
                {view === 'both' ? 'Calibrated (solid) · Raw (light)' : view === 'raw' ? 'Raw' : 'Calibrated'}
              </th>
              {showCalibrated && (
                <th scope="col" className="w-20 py-2 text-right font-medium">
                  {compact ? 'Probability' : 'Calibrated'}
                </th>
              )}
              {showRaw && (
                <th scope="col" className="w-20 py-2 pl-2 text-right font-medium">
                  Raw
                </th>
              )}
              {!compact && hasThresholds && (
                <th scope="col" className="w-28 py-2 pl-2 text-right font-medium print:table-cell">
                  {view === 'both' ? 'Threshold (Cal · Raw)' : view === 'raw' ? 'Threshold (Raw)' : 'Threshold (Cal)'}
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {sortedPredictions.map((p, index) => {
              const calThr = p.calibratedThreshold ?? p.threshold;
              const rawThr = p.rawThreshold ?? p.threshold;
              const isCalPos =
                p.calibratedFlagged !== undefined && p.calibratedFlagged !== null
                  ? p.calibratedFlagged
                  : p.calibratedProbability !== null && calThr !== null
                  ? p.calibratedProbability >= calThr
                  : p.flagged;
              const isRawPos =
                p.rawFlagged !== undefined && p.rawFlagged !== null
                  ? p.rawFlagged
                  : p.rawProbability !== null && rawThr !== null
                  ? p.rawProbability >= rawThr
                  : p.flagged;

              const activeFlag = view === 'raw' ? isRawPos : isCalPos;
              const activeThr = view === 'raw' ? rawThr : calThr;

              return (
                <tr key={p.disease} className="border-b border-line/70 last:border-0">
                  <td className="tabular py-3 pr-2 text-xs text-muted">{String(index + 1).padStart(2, '0')}</td>
                  <th scope="row" className="py-3 pr-4 text-left font-medium">
                    <div className="flex items-center gap-2">
                      <span>{formatDiseaseName(p.disease)}</span>
                      {sortBy === 'highest' && (
                        <span className="rounded bg-subtle px-1.5 py-0.5 text-[0.6875rem] font-normal text-muted tabular">
                          Rank #{index + 1}
                        </span>
                      )}
                    </div>
                    {/* Clear percentage of disease and decision threshold below finding name */}
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs font-normal">
                      <span className="font-semibold text-ink">
                        {p.calibratedProbability !== null ? (
                          <>
                            {formatPercent(p.calibratedProbability)}
                            {view === 'both' && p.rawProbability !== null && (
                              <span className="font-normal text-muted"> ({formatPercent(p.rawProbability)} raw)</span>
                            )}
                          </>
                        ) : p.rawProbability !== null ? (
                          formatPercent(p.rawProbability)
                        ) : (
                          <span className="text-muted">Not provided</span>
                        )}
                      </span>
                      {!compact && activeFlag !== null && (
                        <span
                          className={cn(
                            'inline-flex items-center rounded px-1.5 py-0.5 text-[0.6875rem] font-medium',
                            activeFlag
                              ? 'bg-amber-500/10 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300'
                              : 'bg-subtle text-muted',
                          )}
                        >
                          {activeFlag ? 'Above threshold' : 'Below threshold'}
                          {activeThr !== null && activeThr !== undefined ? ` (${formatPercent(activeThr)})` : ''}
                        </span>
                      )}
                    </div>
                  </th>
                  <td className="hidden py-3 pr-4 sm:table-cell">
                    <div className="flex min-w-[8rem] flex-col gap-1.5">
                      {showCalibrated && (
                        <Bar
                          value={p.calibratedProbability}
                          tone="solid"
                          threshold={compact ? null : (p.calibratedThreshold ?? p.threshold)}
                        />
                      )}
                      {showRaw && (
                        <Bar
                          value={p.rawProbability}
                          tone={view === 'both' ? 'outline' : 'solid'}
                          threshold={compact ? null : (p.rawThreshold ?? p.threshold)}
                        />
                      )}
                    </div>
                  </td>
                  {showCalibrated && (
                    <td className="tabular py-3 text-right font-medium">
                      {p.calibratedProbability === null ? (
                        <span className="text-xs font-normal text-muted">Not provided</span>
                      ) : (
                        formatPercent(p.calibratedProbability)
                      )}
                    </td>
                  )}
                  {showRaw && (
                    <td className={cn('tabular py-3 pl-2 text-right', view === 'both' ? 'text-muted' : 'font-medium')}>
                      {p.rawProbability === null ? (
                        <span className="text-xs font-normal text-muted">Not provided</span>
                      ) : (
                        formatPercent(p.rawProbability)
                      )}
                    </td>
                  )}
                  {!compact && hasThresholds && (
                    <td className="tabular py-3 pl-2 text-right font-medium text-muted print:table-cell">
                      {view === 'both' ? (
                        <span>
                          <span className="font-semibold text-ink">
                            {p.calibratedThreshold !== null && p.calibratedThreshold !== undefined
                              ? formatPercent(p.calibratedThreshold)
                              : '—'}
                          </span>
                          <span className="text-xs text-muted">
                            {' · '}
                            {p.rawThreshold !== null && p.rawThreshold !== undefined
                              ? formatPercent(p.rawThreshold)
                              : formatPercent(p.threshold ?? 0)}
                          </span>
                        </span>
                      ) : activeThr === null || activeThr === undefined ? (
                        '—'
                      ) : (
                        formatPercent(activeThr)
                      )}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {!compact && (
        <div className="mt-6 rounded-lg border border-line bg-subtle/30 p-4">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Findings Breakdown (Ranked Highest to Lowest)
            </h4>
            <span className="text-xs text-muted tabular">
              {sortedPredictions.filter((p) => {
                const calThr = p.calibratedThreshold ?? p.threshold;
                const rawThr = p.rawThreshold ?? p.threshold;
                const isCalPos =
                  p.calibratedFlagged !== undefined && p.calibratedFlagged !== null
                    ? p.calibratedFlagged
                    : p.calibratedProbability !== null && calThr !== null
                    ? p.calibratedProbability >= calThr
                    : p.flagged;
                const isRawPos =
                  p.rawFlagged !== undefined && p.rawFlagged !== null
                    ? p.rawFlagged
                    : p.rawProbability !== null && rawThr !== null
                    ? p.rawProbability >= rawThr
                    : p.flagged;
                return view === 'raw' ? isRawPos : isCalPos;
              }).length} positive finding(s)
            </span>
          </div>

          <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {sortedPredictions.map((p, idx) => {
              const calThr = p.calibratedThreshold ?? p.threshold;
              const rawThr = p.rawThreshold ?? p.threshold;
              const isCalPos =
                p.calibratedFlagged !== undefined && p.calibratedFlagged !== null
                  ? p.calibratedFlagged
                  : p.calibratedProbability !== null && calThr !== null
                  ? p.calibratedProbability >= calThr
                  : p.flagged;
              const isRawPos =
                p.rawFlagged !== undefined && p.rawFlagged !== null
                  ? p.rawFlagged
                  : p.rawProbability !== null && rawThr !== null
                  ? p.rawProbability >= rawThr
                  : p.flagged;
              const activeFlag = view === 'raw' ? isRawPos : isCalPos;
              const activeThr = view === 'raw' ? rawThr : calThr;
              const displayProb =
                view === 'raw'
                  ? p.rawProbability
                  : (p.calibratedProbability ?? p.rawProbability);

              return (
                <div
                  key={p.disease}
                  className={cn(
                    'flex items-center justify-between rounded-md border px-3 py-2 text-xs transition-colors',
                    activeFlag
                      ? 'border-amber-500/40 bg-amber-500/5 dark:bg-amber-500/10'
                      : 'border-line/60 bg-surface',
                  )}
                >
                  <div className="min-w-0 pr-2">
                    <div className="flex items-center gap-1.5 font-medium text-ink">
                      <span className="tabular text-[0.6875rem] text-muted">
                        #{String(idx + 1).padStart(2, '0')}
                      </span>
                      <span className="truncate">{formatDiseaseName(p.disease)}</span>
                    </div>
                    <span className="text-[0.6875rem] text-muted">
                      {activeFlag ? 'Above threshold' : 'Below threshold'}
                      {activeThr !== null && activeThr !== undefined ? ` (${formatPercent(activeThr)})` : ''}
                    </span>
                  </div>
                  <div className="text-right">
                    <span
                      className={cn(
                        'tabular font-semibold',
                        activeFlag ? 'text-amber-700 dark:text-amber-300' : 'text-ink',
                      )}
                    >
                      {displayProb !== null && displayProb !== undefined
                        ? formatPercent(displayProb)
                        : '—'}
                    </span>
                    {view === 'both' && p.rawProbability !== null && p.calibratedProbability !== null && (
                      <span className="block text-[0.625rem] text-muted">
                        {formatPercent(p.rawProbability)} raw
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

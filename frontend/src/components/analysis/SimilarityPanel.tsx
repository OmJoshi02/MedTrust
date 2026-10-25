import { useState, useEffect, useMemo } from 'react';
import {
  Activity,
  CircleAlert,
  Database,
  FileText,
  Info,
  Layers,
  ChevronDown,
} from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Dialog } from '@/components/ui/Dialog';
import { formatNumber } from '@/lib/format';
import type {
  PredictedFindingEvidence,
  ReferenceReport,
  SimilarityAssessment,
  SimilarityMatch,
} from '@/types/analysis';
import { Unavailable } from './SectionShell';

interface SimilarityPanelProps {
  similarity: SimilarityAssessment | undefined;
  queryImageUrl?: string;
  queryFilename?: string;
  predictedFindings?: string[];
  analysisId?: string;
  onKChange?: (k: number) => void;
}

const CANONICAL_NIH_PREVALENCE_MAP: Record<string, number> = {
  Infiltration: 17.74,
  Effusion: 11.88,
  Atelectasis: 10.31,
  Nodule: 5.65,
  Mass: 5.16,
  Pneumothorax: 4.73,
  Consolidation: 4.16,
  Pleural_Thickening: 3.02,
  Cardiomegaly: 2.48,
  Emphysema: 2.24,
  Edema: 2.05,
  Fibrosis: 1.50,
  Pneumonia: 1.28,
  Hernia: 0.20,
  'No Finding': 53.84,
};

export function SimilarityPanel({
  similarity,
  predictedFindings,
  onKChange,
}: SimilarityPanelProps) {
  const [inspectingCase, setInspectingCase] = useState<string | null>(null);
  const [reportData, setReportData] = useState<ReferenceReport | null>(null);
  const [loadingReport, setLoadingReport] = useState<boolean>(false);
  const [reportError, setReportError] = useState<string | null>(null);

  // Initialize selectedK based on similarity.top_k, or matches length, or 5
  const initialK = similarity?.top_k ?? (similarity?.matches ? Math.min(similarity.matches.length, 5) : 5);
  const [selectedK, setSelectedK] = useState<number>(initialK > 0 ? initialK : 5);

  const handleKChange = (newK: number) => {
    setSelectedK(newK);
    if (onKChange) {
      onKChange(newK);
    }
  };

  useEffect(() => {
    if (!inspectingCase) {
      setReportData(null);
      setReportError(null);
      return;
    }
    let cancelled = false;
    setLoadingReport(true);
    setReportError(null);

    fetch(`/api/similarity/cases/${encodeURIComponent(inspectingCase)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Report HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setReportData(data);
      })
      .catch((err) => {
        if (!cancelled) setReportError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoadingReport(false);
      });

    return () => {
      cancelled = true;
    };
  }, [inspectingCase]);

  if (!similarity || !similarity.matches || similarity.matches.length === 0) {
    return (
      <Unavailable>
        Case similarity analysis was not returned for this analysis. It was not evaluated.
      </Unavailable>
    );
  }

  const allMatches = similarity.matches;
  const candidatesSearched =
    similarity.candidates_searched ?? similarity.total_references ?? 2500;

  // Sliced matches based on selectedK
  const effectiveMatches = useMemo(() => {
    return allMatches.slice(0, selectedK);
  }, [allMatches, selectedK]);

  const effectiveK = effectiveMatches.length;
  const isDefaultK = selectedK === (similarity.top_k ?? allMatches.length);

  // Resolve target predicted findings
  const targetFindings = useMemo(() => {
    const rawTargetFindings =
      similarity.predicted_findings && similarity.predicted_findings.length > 0
        ? similarity.predicted_findings
        : predictedFindings && predictedFindings.length > 0
        ? predictedFindings
        : similarity.primary_query_finding
        ? [similarity.primary_query_finding]
        : ['No Finding'];

    const unique = Array.from(new Set(rawTargetFindings.filter(Boolean)));
    return unique.length > 0 ? unique : ['No Finding'];
  }, [similarity.predicted_findings, similarity.primary_query_finding, predictedFindings]);

  const primaryFinding =
    similarity.primary_query_finding ||
    (targetFindings.length > 0 && targetFindings[0] !== 'No Finding' ? targetFindings[0] : undefined);

  // Dynamically compute evidence list for effectiveMatches
  const evidenceList: PredictedFindingEvidence[] = useMemo(() => {
    // If on default K and backend provided predicted_findings_evidence, preserve authoritative backend records
    if (isDefaultK && similarity.predicted_findings_evidence && similarity.predicted_findings_evidence.length > 0) {
      return similarity.predicted_findings_evidence;
    }

    const weights = effectiveMatches.map((m) => Math.max(0, m.similarity_score));
    const totalWeight = weights.reduce((acc, w) => acc + w, 0);

    return targetFindings.map((finding) => {
      const isNormal =
        finding.toLowerCase() === 'no finding' || finding.toLowerCase() === 'normal';

      const matchInSupport = isDefaultK
        ? similarity.weighted_label_support?.find(
            (s) =>
              s.disease.toLowerCase() === finding.toLowerCase() ||
              (s.disease_name && s.disease_name.toLowerCase() === finding.toLowerCase()),
          )
        : undefined;

      const matchingRefs: SimilarityMatch[] = [];
      const matchingIndices: number[] = [];

      effectiveMatches.forEach((m, idx) => {
        const rawFindings = m.findings || [];
        const labelStr = (m.finding_labels || m.metadata?.finding_labels || '').toLowerCase();
        const fList = rawFindings.map((f) => String(f).toLowerCase());
        const shared = (m.concordance?.shared_findings || []).map((f) => String(f).toLowerCase());
        const refFindings = (m.concordance?.reference_findings || []).map((f) =>
          String(f).toLowerCase(),
        );

        let matchesDisease = false;
        if (isNormal) {
          matchesDisease =
            (fList.length === 0 && !labelStr) ||
            fList.includes('no finding') ||
            labelStr.includes('no finding');
        } else {
          const norm = finding.toLowerCase().replace(/_/g, ' ');
          const rawNorm = finding.toLowerCase();
          matchesDisease =
            fList.includes(norm) ||
            fList.includes(rawNorm) ||
            labelStr.includes(norm) ||
            labelStr.includes(rawNorm) ||
            shared.includes(norm) ||
            shared.includes(rawNorm) ||
            refFindings.includes(norm) ||
            refFindings.includes(rawNorm) ||
            (Boolean(m.concordance?.matches_primary) &&
              primaryFinding?.toLowerCase() === norm);
        }

        if (matchesDisease) {
          matchingIndices.push(idx);
          matchingRefs.push(m);
        }
      });

      const matchingCases = matchInSupport
        ? matchInSupport.retrieved_count
        : matchingRefs.length;
      const retrievedCases = matchInSupport?.retrieved_denominator ?? effectiveK;
      const matchingRate =
        matchInSupport?.retrieval_rate ??
        (retrievedCases > 0 ? (matchingCases / retrievedCases) * 100 : 0);

      let weightedSupport: number | null = null;
      if (matchInSupport && matchInSupport.weighted_support !== null) {
        weightedSupport = matchInSupport.weighted_support;
      } else if (totalWeight > 0 && matchingIndices.length > 0) {
        const weightedSum = matchingIndices.reduce((acc, idx) => acc + weights[idx], 0);
        weightedSupport = (weightedSum / totalWeight) * 100;
      } else {
        weightedSupport = 0;
      }

      const datasetPrev =
        matchInSupport?.dataset_prevalence ??
        CANONICAL_NIH_PREVALENCE_MAP[finding] ??
        CANONICAL_NIH_PREVALENCE_MAP[finding.replace(/_/g, ' ')] ??
        null;

      let explanation = '';
      if (matchingCases === 0) {
        explanation = `0 of ${retrievedCases} retrieved reference cases (0.0% matching rate, 0.0% weighted support) match predicted ${finding}. No reference support found in top matches.`;
      } else {
        explanation = `${matchingCases} of ${retrievedCases} retrieved reference cases (${matchingRate.toFixed(1)}% matching rate, ${weightedSupport?.toFixed(1) ?? '0.0'}% weighted support) corroborate predicted ${finding}.`;
      }

      return {
        predicted_finding: finding,
        matching_cases: matchingCases,
        retrieved_cases: retrievedCases,
        matching_rate: matchingRate,
        weighted_support: weightedSupport ?? 0,
        dataset_prevalence: datasetPrev ?? undefined,
        is_matched: matchingCases > 0,
        matching_references: matchingRefs,
        explanation,
      };
    });
  }, [isDefaultK, similarity.predicted_findings_evidence, similarity.weighted_label_support, effectiveMatches, effectiveK, targetFindings, primaryFinding]);

  // Top match calculation
  const topMatch = effectiveMatches[0] ?? similarity.top_match ?? null;
  const topSimilarityScore = topMatch && typeof topMatch.similarity_score === 'number'
    ? topMatch.similarity_score
    : null;

  // Primary Concordance calculation
  const { primaryConcordantCount, primaryConcordanceRate } = useMemo(() => {
    if (isDefaultK && typeof similarity.primary_concordance_rate === 'number' && typeof similarity.primary_concordant_count === 'number') {
      return {
        primaryConcordantCount: similarity.primary_concordant_count,
        primaryConcordanceRate: similarity.primary_concordance_rate,
      };
    }
    if (!primaryFinding || effectiveK === 0) {
      return { primaryConcordantCount: 0, primaryConcordanceRate: 0 };
    }
    const count = effectiveMatches.filter((m) => {
      if (m.concordance?.matches_primary !== undefined) {
        return m.concordance.matches_primary;
      }
      const raw = (m.findings || []).map((f) => String(f).toLowerCase());
      const labelStr = (m.finding_labels || m.metadata?.finding_labels || '').toLowerCase();
      const norm = primaryFinding.toLowerCase().replace(/_/g, ' ');
      return raw.includes(norm) || labelStr.includes(norm);
    }).length;

    return {
      primaryConcordantCount: count,
      primaryConcordanceRate: (count / effectiveK) * 100,
    };
  }, [isDefaultK, similarity.primary_concordance_rate, similarity.primary_concordant_count, primaryFinding, effectiveK, effectiveMatches]);

  // Any-Finding Overlap calculation
  const { anyFindingOverlapCount, anyFindingOverlapRate } = useMemo(() => {
    if (isDefaultK && typeof similarity.any_finding_overlap_rate === 'number' && typeof similarity.any_finding_overlap_count === 'number') {
      return {
        anyFindingOverlapCount: similarity.any_finding_overlap_count,
        anyFindingOverlapRate: similarity.any_finding_overlap_rate,
      };
    }
    if (effectiveK === 0) {
      return { anyFindingOverlapCount: 0, anyFindingOverlapRate: 0 };
    }
    const count = effectiveMatches.filter((m) => {
      if (m.concordance?.has_overlap !== undefined) {
        return m.concordance.has_overlap;
      }
      const raw = (m.findings || []).map((f) => String(f).toLowerCase());
      const labelStr = (m.finding_labels || m.metadata?.finding_labels || '').toLowerCase();
      return targetFindings.some((tf) => {
        const norm = tf.toLowerCase().replace(/_/g, ' ');
        return raw.includes(norm) || labelStr.includes(norm);
      });
    }).length;

    return {
      anyFindingOverlapCount: count,
      anyFindingOverlapRate: (count / effectiveK) * 100,
    };
  }, [isDefaultK, similarity.any_finding_overlap_rate, similarity.any_finding_overlap_count, effectiveK, effectiveMatches, targetFindings]);

  // Dynamic concordance summary
  const dynamicConcordanceSummary = useMemo(() => {
    if (isDefaultK && similarity.concordance_summary) {
      return similarity.concordance_summary;
    }
    if (primaryFinding) {
      return `${primaryConcordantCount} of ${effectiveK} (${primaryConcordanceRate.toFixed(1)}%) retrieved reference cases corroborate primary ${primaryFinding} finding; ${anyFindingOverlapCount} of ${effectiveK} (${anyFindingOverlapRate.toFixed(1)}%) share any detected finding.`;
    }
    return `${anyFindingOverlapCount} of ${effectiveK} (${anyFindingOverlapRate.toFixed(1)}%) retrieved reference cases share detected findings.`;
  }, [isDefaultK, similarity.concordance_summary, primaryFinding, primaryConcordantCount, effectiveK, primaryConcordanceRate, anyFindingOverlapCount, anyFindingOverlapRate]);

  return (
    <div className="space-y-6">
      {/* Reference Intelligence Library Context & Top-K Retrieval Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-line bg-surface px-4 py-3 shadow-xs">
        <div className="flex items-center gap-2.5">
          <Database className="h-4 w-4 text-charcoal shrink-0" />
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-ink">Reference Case Intelligence</span>
              <Badge tone="dark">Active Index</Badge>
            </div>
            <p className="text-[11px] text-muted">
              Searched <strong className="font-semibold text-ink">{candidatesSearched.toLocaleString()}</strong> reference cases · DenseNet-121 (1024-d)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted font-medium">Retrieval Depth:</span>
          <div
            role="radiogroup"
            aria-label="Top-K retrieval depth"
            className="inline-flex rounded-md border border-line bg-subtle/50 p-0.5"
          >
            {[5, 10, 20].map((kOption) => {
              const isSelected = selectedK === kOption;
              return (
                <button
                  key={kOption}
                  type="button"
                  role="radio"
                  aria-checked={isSelected}
                  onClick={() => handleKChange(kOption)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-charcoal ${
                    isSelected
                      ? 'bg-ink text-white shadow-xs'
                      : 'text-muted hover:text-ink hover:bg-subtle'
                  }`}
                >
                  Top-{kOption}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Notice if requested K exceeds stored matches */}
      {selectedK > allMatches.length && (
        <div className="rounded border border-line bg-subtle/60 px-3.5 py-2 text-xs text-muted flex items-center gap-2">
          <Info className="h-4 w-4 text-charcoal shrink-0" />
          <span>
            Current report contains {allMatches.length} retrieved reference cases. Displaying all {allMatches.length} available cases (requested depth: Top-{selectedK}).
          </span>
        </div>
      )}

      {/* 3 Summary Metric Cards */}
      <dl className="grid gap-4 sm:grid-cols-3">
        {/* Card 1: Top-Match Similarity */}
        <div className="flex flex-col justify-between rounded-lg border border-line bg-surface p-4 shadow-xs">
          <div>
            <div className="flex items-center justify-between">
              <dt className="text-xs font-medium text-muted">Top Match Alignment</dt>
              <Badge tone="neutral">Cosine similarity</Badge>
            </div>
            <dd className="mt-2 flex items-baseline gap-2">
              <span className="tabular text-2xl font-semibold tracking-tight text-ink">
                {topSimilarityScore !== null
                  ? `${formatNumber(topSimilarityScore, 1)}%`
                  : 'N/A'}
              </span>
              {topSimilarityScore !== null && (
                <Badge tone={topSimilarityScore >= 80 ? 'success' : 'neutral'}>
                  {topSimilarityScore >= 80 ? 'High' : 'Moderate'}
                </Badge>
              )}
            </dd>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-muted border-t border-line/40 pt-2">
            Highest visual embedding similarity between query and retrieved reference cases.
          </p>
        </div>

        {/* Card 2: Primary Finding Concordance */}
        <div className="flex flex-col justify-between rounded-lg border border-line bg-surface p-4 shadow-xs">
          <div>
            <div className="flex items-center justify-between">
              <dt className="text-xs font-medium text-muted">Primary Concordance (C_d)</dt>
              <Badge
                tone={primaryConcordanceRate >= 50 ? 'success' : 'warning'}
              >
                {primaryConcordantCount} / {effectiveK}
              </Badge>
            </div>
            <dd className="mt-2 flex items-baseline gap-2">
              <span className="tabular text-2xl font-semibold tracking-tight text-ink">
                {formatNumber(primaryConcordanceRate, 1)}%
              </span>
              {primaryFinding && (
                <span className="text-[11px] font-medium text-muted truncate max-w-[120px]" title={primaryFinding}>
                  {primaryFinding} match
                </span>
              )}
            </dd>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-muted border-t border-line/40 pt-2">
            Retrieved reference cases sharing the primary finding ({primaryFinding || 'primary'}).
          </p>
        </div>

        {/* Card 3: Any-Finding Overlap */}
        <div className="flex flex-col justify-between rounded-lg border border-line bg-surface p-4 shadow-xs">
          <div>
            <div className="flex items-center justify-between">
              <dt className="text-xs font-medium text-muted">Any-Finding Overlap (O)</dt>
              <Badge
                tone={anyFindingOverlapRate >= 50 ? 'success' : 'neutral'}
              >
                {anyFindingOverlapCount} / {effectiveK}
              </Badge>
            </div>
            <dd className="mt-2 flex items-baseline gap-2">
              <span className="tabular text-2xl font-semibold tracking-tight text-ink">
                {formatNumber(anyFindingOverlapRate, 1)}%
              </span>
              <span className="text-[11px] font-medium text-muted">
                Multilabel concordance
              </span>
            </dd>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-muted border-t border-line/40 pt-2">
            Retrieved reference cases sharing at least one finding with the query prediction.
          </p>
        </div>
      </dl>

      {/* Diagnostic Concordance Assessment Banner */}
      {dynamicConcordanceSummary && (
        <div
          className={`flex items-start gap-3 rounded-lg border p-3.5 text-xs transition-colors ${
            primaryConcordanceRate >= 60
              ? 'border-emerald-200 bg-emerald-50/60 text-emerald-950'
              : 'border-line bg-subtle/80 text-ink'
          }`}
        >
          <Activity className="mt-0.5 h-4 w-4 shrink-0 text-charcoal" />
          <div className="flex-1 space-y-1">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-semibold text-ink">
                Diagnostic Concordance Assessment:
              </span>
              {primaryFinding && (
                <Badge tone="dark">Primary: {primaryFinding}</Badge>
              )}
            </div>
            <p className="text-muted leading-relaxed">{dynamicConcordanceSummary}</p>
          </div>
        </div>
      )}

      {/* Methodological Explanation Collapsible Section */}
      <details className="group rounded-lg border border-line bg-surface p-4 shadow-xs">
        <summary className="cursor-pointer text-xs font-semibold text-ink flex items-center justify-between select-none list-none">
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-charcoal shrink-0" />
            <span>How to Interpret Case-Based Similarity Metrics</span>
          </div>
          <div className="flex items-center gap-1 text-[11px] text-muted font-normal">
            <span className="group-open:hidden">Show definitions</span>
            <span className="hidden group-open:inline">Hide definitions</span>
            <ChevronDown className="h-3.5 w-3.5 text-charcoal transition-transform duration-200 group-open:rotate-180" />
          </div>
        </summary>
        <div className="mt-3.5 pt-3 border-t border-line/60 space-y-3.5 text-xs text-muted leading-relaxed">
          <div className="grid gap-3.5 sm:grid-cols-2">
            <div className="rounded border border-line/40 bg-subtle/30 p-2.5">
              <h5 className="font-semibold text-ink text-[11px]">Feature Cosine Similarity</h5>
              <p className="mt-0.5 text-[11px]">
                Measures vector alignment between DenseNet-121 1024-dimensional feature embeddings. Scores reflect visual representation alignment in deep feature space (0%–100%).
              </p>
            </div>
            <div className="rounded border border-line/40 bg-subtle/30 p-2.5">
              <h5 className="font-semibold text-ink text-[11px]">Primary Concordance Rate (C_d)</h5>
              <p className="mt-0.5 text-[11px]">
                Proportion of top-K retrieved cases whose verified NIH ground truth labels match the model&apos;s primary predicted finding.
              </p>
            </div>
            <div className="rounded border border-line/40 bg-subtle/30 p-2.5">
              <h5 className="font-semibold text-ink text-[11px]">Multilabel Any-Finding Overlap (O)</h5>
              <p className="mt-0.5 text-[11px]">
                Proportion of top-K retrieved cases that share at least one detected finding with the query radiograph&apos;s positive findings.
              </p>
            </div>
            <div className="rounded border border-line/40 bg-subtle/30 p-2.5">
              <h5 className="font-semibold text-ink text-[11px]">Retrieval Match Rate (R_d)</h5>
              <p className="mt-0.5 text-[11px]">
                Fraction of retrieved cases presenting finding <em>d</em> (K_d / K). Indicates empirical frequency of the disease among visually similar cases.
              </p>
            </div>
            <div className="rounded border border-line/40 bg-subtle/30 p-2.5">
              <h5 className="font-semibold text-ink text-[11px]">Cosine-Weighted Support (V_d)</h5>
              <p className="mt-0.5 text-[11px]">
                Proportion of cosine similarity allocated to cases presenting finding <em>d</em>, giving greater weight to more visually similar reference cases.
              </p>
            </div>
            <div className="rounded border border-line/40 bg-subtle/30 p-2.5">
              <h5 className="font-semibold text-ink text-[11px]">Baseline Population Prevalence (P_d)</h5>
              <p className="mt-0.5 text-[11px]">
                Background frequency of finding <em>d</em> across all 112,120 radiographs in the NIH ChestX-ray14 dataset.
              </p>
            </div>
          </div>
          <div className="pt-2 text-[11px] border-t border-line/40 text-muted">
            <span className="font-semibold text-ink">Clinical Disclaimer: </span>
            Case-based similarity retrieval provides observational context from historical reference cases. Visual similarity in deep feature space does not constitute a diagnostic confirmation or substitute for qualified radiological evaluation.
          </div>
        </div>
      </details>

      {/* PREDICTION-DRIVEN REFERENCE EVIDENCE SECTION (Streamlined progress bars without bulky blocks) */}
      <div className="rounded-lg border border-line bg-surface p-4 shadow-xs space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line/60 pb-3">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-charcoal shrink-0" />
            <div>
              <h4 className="text-xs font-semibold text-ink">
                Prediction-Driven Reference Evidence ({evidenceList.length}{' '}
                {evidenceList.length === 1 ? 'predicted finding' : 'predicted findings'})
              </h4>
              <p className="text-[11px] text-muted">
                Reference case retrieval concordance across top-{effectiveK} similar reference radiographs.
              </p>
            </div>
          </div>
          <Badge tone="neutral">Top-{effectiveK} Retrieval</Badge>
        </div>

        {/* Streamlined progress bar list for every finding */}
        <div className="divide-y divide-line/40">
          {evidenceList.map((item) => {
            const hasMatches = item.matching_cases > 0;
            const refCases = item.matching_references || [];

            return (
              <div
                key={item.predicted_finding}
                className="py-3 space-y-2 first:pt-1 last:pb-1"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  {/* Finding Label and Match Badge */}
                  <div className="flex items-center gap-2 min-w-[200px]">
                    <span className="text-xs font-semibold text-ink">
                      {item.predicted_finding}
                    </span>
                    <Badge tone={hasMatches ? 'success' : 'neutral'}>
                      {hasMatches
                        ? `${item.matching_cases} of ${item.retrieved_cases} matches`
                        : `0 of ${item.retrieved_cases} matches`}
                    </Badge>
                  </div>

                  {/* Horizontal Evidence Progress Bar & Percentage */}
                  <div className="flex-1 flex items-center gap-3">
                    <div
                      role="progressbar"
                      aria-valuenow={item.matching_rate}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-label={`Matching retrieval rate for ${item.predicted_finding}: ${formatNumber(item.matching_rate, 1)}%`}
                      aria-valuetext={`${formatNumber(item.matching_rate, 1)}%`}
                      className="h-2.5 flex-1 overflow-hidden rounded-full bg-subtle border border-line/50"
                    >
                      <div
                        className={`h-full transition-all duration-300 ${
                          hasMatches ? 'bg-emerald-600' : 'bg-muted/30'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, item.matching_rate))}%` }}
                      />
                    </div>
                    <span className="tabular font-semibold text-xs text-ink w-14 text-right">
                      {formatNumber(item.matching_rate, 1)}%
                    </span>
                  </div>

                  {/* Case inspection toggle if matching cases exist */}
                  {hasMatches && refCases.length > 0 && (
                    <details className="group shrink-0 sm:ml-2">
                      <summary className="cursor-pointer inline-flex items-center gap-1 text-[11px] font-medium text-charcoal hover:underline select-none list-none">
                        <FileText className="h-3.5 w-3.5 text-charcoal" />
                        <span>Inspect cases ({refCases.length})</span>
                      </summary>
                      <div className="mt-2.5 overflow-x-auto rounded border border-line bg-surface">
                        <div className="min-w-[540px]">
                          <div className="grid grid-cols-12 border-b border-line/60 bg-subtle/40 py-1.5 px-3 text-[11px] font-medium text-muted">
                            <div className="col-span-1">Rank</div>
                            <div className="col-span-3">Image ID</div>
                            <div className="col-span-2">Similarity</div>
                            <div className="col-span-2">Demographics</div>
                            <div className="col-span-3">Verified Ground Truth</div>
                            <div className="col-span-1 text-right">Action</div>
                          </div>
                          <div className="divide-y divide-line/30 text-xs">
                            {refCases.map((ref) => {
                              const meta = ref.metadata || {};
                              const age = meta.patient_age;
                              const gender = meta.patient_gender;
                              const view = meta.view_position;
                              const demoParts = [
                                age ? `${age}y` : null,
                                gender && gender !== 'U' ? gender : null,
                                view,
                              ].filter(Boolean);
                              const demoStr = demoParts.length > 0 ? demoParts.join(' · ') : 'N/A';

                              const fList =
                                ref.findings && ref.findings.length > 0
                                  ? ref.findings.join(' | ')
                                  : ref.finding_labels || meta.finding_labels || 'No Finding';

                              return (
                                <div
                                  key={`${item.predicted_finding}-${ref.rank}-${ref.filename}`}
                                  className="grid grid-cols-12 py-2 px-3 items-center hover:bg-subtle/30 transition-colors"
                                >
                                  <div className="col-span-1 tabular text-muted">#{ref.rank}</div>
                                  <div
                                    className="col-span-3 font-mono text-[11px] text-ink truncate cursor-pointer hover:underline hover:text-charcoal"
                                    title={`Click to inspect report for ${ref.filename}`}
                                    onClick={() => setInspectingCase(ref.filename)}
                                  >
                                    {ref.filename}
                                  </div>
                                  <div className="col-span-2 tabular font-semibold text-ink">
                                    {formatNumber(ref.similarity_score, 1)}%
                                  </div>
                                  <div className="col-span-2 text-muted text-[11px]">{demoStr}</div>
                                  <div
                                    className="col-span-3 text-muted text-[11px] truncate"
                                    title={fList}
                                  >
                                    {fList}
                                  </div>
                                  <div className="col-span-1 text-right">
                                    <button
                                      type="button"
                                      onClick={() => setInspectingCase(ref.filename)}
                                      className="inline-flex items-center gap-1 text-[11px] text-charcoal hover:underline font-medium"
                                      title={`Inspect intelligence report for ${ref.filename}`}
                                    >
                                      <FileText className="h-3 w-3" />
                                      <span>Report</span>
                                    </button>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </details>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Medical disclaimer note */}
      <p className="flex items-start gap-2 rounded-md border border-line bg-subtle/60 p-3.5 text-xs leading-relaxed text-muted">
        <CircleAlert className="mt-0.5 h-4 w-4 shrink-0 text-charcoal" />
        <span>{similarity.disclaimer}</span>
      </p>

      {/* Reference Case Intelligence Inspector Dialog */}
      <Dialog
        open={inspectingCase !== null}
        onClose={() => setInspectingCase(null)}
        title={inspectingCase ? `Reference Case Intelligence — ${inspectingCase}` : 'Case Intelligence'}
        description="Precomputed multi-module intelligence report from frozen MedTrust evaluation pipeline."
        size="xl"
      >
        <div className="space-y-4 text-xs">
          {loadingReport && (
            <div className="py-8 text-center text-muted">
              Loading reference case report...
            </div>
          )}

          {reportError && (
            <div className="rounded border border-red-200 bg-red-50 p-3 text-red-700">
              Failed to load reference case report: {reportError}
            </div>
          )}

          {reportData && (
            <div className="space-y-4">
              {/* Demographics & Provenance */}
              <div className="rounded-lg border border-line bg-surface p-3.5 space-y-2">
                <div className="flex items-center justify-between border-b border-line/60 pb-2">
                  <span className="font-semibold text-ink">Patient & Acquisition Metadata</span>
                  <Badge tone="neutral">NIH ChestX-ray14 ({reportData.dataset_split || 'train'})</Badge>
                </div>
                <dl className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <div>
                    <dt className="text-muted text-[11px]">Patient ID</dt>
                    <dd className="font-mono text-ink font-semibold">{reportData.patient_id || 'N/A'}</dd>
                  </div>
                  <div>
                    <dt className="text-muted text-[11px]">Age</dt>
                    <dd className="text-ink font-semibold">{reportData.demographics?.patient_age ? `${reportData.demographics.patient_age} yrs` : 'N/A'}</dd>
                  </div>
                  <div>
                    <dt className="text-muted text-[11px]">Gender</dt>
                    <dd className="text-ink font-semibold">{reportData.demographics?.patient_gender || 'N/A'}</dd>
                  </div>
                  <div>
                    <dt className="text-muted text-[11px]">View Position</dt>
                    <dd className="text-ink font-semibold">{reportData.demographics?.view_position || 'PA'}</dd>
                  </div>
                </dl>
              </div>

              {/* Ground Truth Dataset Annotations */}
              <div className="rounded-lg border border-line bg-surface p-3.5 space-y-2">
                <span className="font-semibold text-ink block">Dataset Ground Truth Annotations</span>
                <div className="flex flex-wrap gap-1.5">
                  {(reportData.dataset_annotations?.findings || ['No Finding']).map((finding) => (
                    <Badge key={finding} tone={finding === 'No Finding' ? 'neutral' : 'dark'}>
                      {finding}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Technical Image Quality & Out-of-Distribution Status */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="rounded-lg border border-line bg-surface p-3 space-y-1.5">
                  <span className="font-semibold text-ink block">Technical Quality (Module 5)</span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-lg font-semibold text-ink">
                      {reportData.image_quality?.composite_score ?? 'N/A'}/100
                    </span>
                    <Badge tone={reportData.image_quality?.suitable ? 'success' : 'warning'}>
                      {reportData.image_quality?.suitable ? 'Nominal Quality' : 'Caution'}
                    </Badge>
                  </div>
                  <div className="text-[11px] text-muted space-x-2">
                    <span>Sharpness: {formatNumber(reportData.image_quality?.sharpness ?? 0, 1)}</span>
                    <span>·</span>
                    <span>Contrast: {formatNumber(reportData.image_quality?.contrast ?? 0, 1)}</span>
                  </div>
                </div>

                <div className="rounded-lg border border-line bg-surface p-3 space-y-1.5">
                  <span className="font-semibold text-ink block">Domain Distribution (Module 7)</span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-lg font-semibold text-ink">
                      {reportData.ood?.status || 'IN_DISTRIBUTION'}
                    </span>
                    <Badge tone={reportData.ood?.is_ood ? 'danger' : 'success'}>
                      {reportData.ood?.is_ood ? 'OOD' : 'In-Distribution'}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-muted">
                    Verified DenseNet feature representation in training manifold.
                  </p>
                </div>
              </div>

              {/* Model Multi-Label Predictions */}
              {reportData.predictions?.calibrated_scores && (
                <div className="rounded-lg border border-line bg-surface p-3.5 space-y-2">
                  <span className="font-semibold text-ink block">Frozen DenseNet-121 Calibrated Predictions</span>
                  <div className="max-h-48 overflow-y-auto divide-y divide-line/40">
                    {Object.entries(reportData.predictions.calibrated_scores).map(([disease, prob]) => {
                      const isPos = reportData.predictions?.decisions?.[disease] ?? false;
                      return (
                        <div key={disease} className="flex items-center justify-between py-1 px-1">
                          <span className="text-ink">{disease}</span>
                          <div className="flex items-center gap-2">
                            <span className="tabular font-mono text-muted">{(Number(prob) * 100).toFixed(1)}%</span>
                            <Badge tone={isPos ? 'warning' : 'neutral'}>
                              {isPos ? 'POSITIVE' : 'Negative'}
                            </Badge>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Medical Disclaimer */}
              <p className="text-[11px] text-muted italic">
                * Precomputed intelligence report archived during baseline reference library construction.
                Exhibited for clinical decision support; not an autonomous diagnostic finding.
              </p>
            </div>
          )}
        </div>
      </Dialog>
    </div>
  );
}

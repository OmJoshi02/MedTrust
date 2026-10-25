/**
 * API response contract, in one place
 * ====================================
 * The backend schema was not supplied, so every response is parsed
 * defensively here and converted to the typed models in src/types. Adapt THIS
 * file (and endpoints.ts) if the backend differs. Nothing is invented: any
 * field that is absent or malformed becomes `undefined`/`null` and the UI
 * shows "not evaluated" or "unavailable".
 *
 * Analysis / report object (snake_case or camelCase accepted):
 *   id | analysis_id                      string, required
 *   status                                queued | processing | completed | failed
 *   created_at                            ISO-8601 timestamp
 *   filename | image.filename             original file name
 *   image { width, height, format, size_bytes, artifact_name, ...metadata }
 *   predictions                           EITHER an array of
 *                                           { disease, raw_probability, calibrated_probability,
 *                                             threshold?, predicted_positive? }
 *                                         OR an object keyed by disease name
 *                                         OR use raw_probabilities / calibrated_probabilities /
 *                                         thresholds maps (or 14-value arrays in canonical order).
 *                                         Probabilities must be in [0, 1]; other values are ignored.
 *   image_quality | quality               { overall_status?, summary?, metrics: [{ key, label?, value,
 *                                           unit?, status?, description? }] } or a flat object of values
 *   uncertainty                           { method?, summary?, metrics: [...], per_disease?: {...} }
 *   ood | out_of_distribution             { status | is_ood, score?, threshold?, method?, explanation? }
 *   calibration                           { method?, ece?, brier?, nll?, reliability: [{ confidence, accuracy, count? }] }
 *   explanation | explainability          { method?, artifacts: [{ name, label?, disease?, caption? }],
 *                                           stability?: { value, metric?, description? },
 *                                           faithfulness?: { value, metric?, description? } }
 *   model { name, version }, warnings[], error
 *
 * Image artifacts are fetched with GET /analyses/:id/artifacts/:name.
 */

import { humanizeKey } from '@/lib/format';
import {
  asArray,
  asBoolean,
  asNumber,
  asRecord,
  asString,
  isRecord,
  normalizeKey,
  pick,
  type UnknownRecord,
} from '@/lib/safe';
import type { Paginated } from '@/types/api';
import {
  DISEASES,
  topPredictions,
  type AnalysisReport,
  type AnalysisStatus,
  type AnalysisSummary,
  type CalibrationAssessment,
  type DiseaseName,
  type DiseasePrediction,
  type ExplanationArtifact,
  type ExplanationAssessment,
  type ExplanationMetric,
  type ImageInfo,
  type MetricStatus,
  type OODAssessment,
  type OODDetector,
  type OODStatus,
  type QualityAssessment,
  type ReliabilityBin,
  type ReportMetric,
  type GlobalDiseaseStatistic,
  type NoFindingStatistic,
  type PredictedFindingEvidence,
  type SimilarityAssessment,
  type SimilarityConcordance,
  type SimilarityMatch,
  type UncertaintyAssessment,
} from '@/types/analysis';
import type { User } from '@/types/auth';
import { ApiError, defaultMessageFor } from './errors';

/* ------------------------------------------------------------------ */
/* Primitives                                                          */
/* ------------------------------------------------------------------ */

const DISEASE_LOOKUP = new Map<string, DiseaseName>(DISEASES.map((d) => [normalizeKey(d), d]));

export function matchDisease(label: unknown): DiseaseName | undefined {
  const text = asString(label);
  return text ? DISEASE_LOOKUP.get(normalizeKey(text)) : undefined;
}

export function normalizeStatus(value: unknown): AnalysisStatus {
  const key = normalizeKey(asString(value) ?? '');
  if (['completed', 'complete', 'done', 'success', 'succeeded', 'finished'].includes(key)) {
    return 'completed';
  }
  if (['queued', 'pending', 'waiting'].includes(key)) return 'queued';
  if (['processing', 'running', 'inprogress', 'started'].includes(key)) return 'processing';
  if (['failed', 'error', 'errored'].includes(key)) return 'failed';
  return 'unknown';
}

function normalizeMetricStatus(value: unknown): MetricStatus | undefined {
  const key = normalizeKey(asString(value) ?? '');
  if (['pass', 'passed', 'good', 'ok', 'acceptable', 'adequate'].includes(key)) return 'good';
  if (['warn', 'warning', 'caution', 'fair', 'borderline', 'marginal'].includes(key)) return 'caution';
  if (['fail', 'failed', 'poor', 'bad', 'inadequate', 'unacceptable'].includes(key)) return 'poor';
  if (key === 'info') return 'info';
  return undefined;
}

function normalizeOODStatus(value: unknown): OODStatus | undefined {
  if (typeof value === 'boolean') return value ? 'out_of_distribution' : 'in_distribution';
  const key = normalizeKey(asString(value) ?? '');
  if (key === '') return undefined;
  if (key.includes('outof') || key === 'ood' || key === 'outlier') return 'out_of_distribution';
  if (key.includes('indistribution') || key === 'id' || key === 'indist' || key === 'normal') {
    return 'in_distribution';
  }
  if (key.includes('uncertain') || key.includes('borderline')) return 'uncertain';
  return undefined;
}

function unwrapEnvelope(raw: unknown): UnknownRecord | undefined {
  const root = asRecord(raw);
  if (!root) return undefined;
  const hasIdentity = pick(root, 'id', 'analysis_id', 'analysisId') !== undefined;
  if (!hasIdentity) {
    for (const key of ['data', 'report', 'analysis', 'result']) {
      const inner = asRecord(root[key]);
      if (inner) return inner;
    }
  }
  return root;
}

function badResponse(): ApiError {
  return new ApiError({ kind: 'bad_response', message: defaultMessageFor('bad_response') });
}

/* ------------------------------------------------------------------ */
/* User                                                                */
/* ------------------------------------------------------------------ */

export function normalizeUser(raw: unknown, isDemo = false): User | null {
  const root = asRecord(raw);
  if (!root) return null;
  const record = asRecord(pick(root, 'user', 'data')) ?? root;
  const email = asString(pick(record, 'email'));
  if (!email) return null;

  const active = asBoolean(pick(record, 'is_active', 'isActive', 'active'));
  const status =
    asString(pick(record, 'status', 'account_status')) ??
    (active === undefined ? undefined : active ? 'active' : 'inactive');

  return {
    id: asString(pick(record, 'id', 'user_id', 'userId', 'uuid')) ?? email,
    email,
    fullName:
      asString(pick(record, 'full_name', 'fullName', 'name', 'display_name')) ??
      email.split('@')[0] ??
      email,
    createdAt: asString(pick(record, 'created_at', 'createdAt', 'joined_at')),
    status,
    role: asString(pick(record, 'role')),
    isDemo,
  };
}

/* ------------------------------------------------------------------ */
/* Predictions                                                         */
/* ------------------------------------------------------------------ */

interface PredictionEntry {
  raw?: number;
  cal?: number;
  thr?: number;
  rawThr?: number;
  calThr?: number;
  rawFlag?: boolean;
  calFlag?: boolean;
  flag?: boolean;
}

function forEachDiseaseValue(value: unknown, callback: (disease: DiseaseName, v: unknown) => void) {
  if (isRecord(value)) {
    for (const [key, v] of Object.entries(value)) {
      const disease = matchDisease(key);
      if (disease) callback(disease, v);
    }
  } else if (Array.isArray(value) && value.length === DISEASES.length) {
    DISEASES.forEach((disease, index) => callback(disease, value[index]));
  }
}

function normalizePredictions(root: UnknownRecord, warnings: string[]): DiseasePrediction[] {
  const table = new Map<DiseaseName, PredictionEntry>();
  const entry = (disease: DiseaseName): PredictionEntry => {
    let existing = table.get(disease);
    if (!existing) {
      existing = {};
      table.set(disease, existing);
    }
    return existing;
  };

  let ignored = 0;
  const probability = (value: unknown): number | undefined => {
    const n = asNumber(value);
    if (n === undefined) return undefined;
    if (n < 0 || n > 1) {
      ignored += 1;
      return undefined;
    }
    return n;
  };

  const applyRecord = (target: PredictionEntry, record: UnknownRecord) => {
    const raw = probability(
      pick(record, 'raw_probability', 'rawProbability', 'raw', 'probability', 'prob', 'score'),
    );
    if (raw !== undefined) target.raw = raw;
    const calibrated = probability(
      pick(record, 'calibrated_probability', 'calibratedProbability', 'calibrated'),
    );
    if (calibrated !== undefined) target.cal = calibrated;
    const threshold = asNumber(pick(record, 'threshold'));
    if (threshold !== undefined) target.thr = threshold;
    const rawThr = asNumber(pick(record, 'raw_threshold', 'rawThreshold'));
    if (rawThr !== undefined) target.rawThr = rawThr;
    const rawFlag = asBoolean(
      pick(record, 'raw_predicted_positive', 'raw_flagged', 'rawFlagged'),
    );
    if (rawFlag !== undefined) target.rawFlag = rawFlag;
    const calFlag = asBoolean(
      pick(record, 'calibrated_predicted_positive', 'calibrated_flagged', 'calibratedFlagged'),
    );
    if (calFlag !== undefined) target.calFlag = calFlag;
    const flag = asBoolean(
      pick(record, 'predicted_positive', 'positive', 'flagged', 'is_positive', 'predicted'),
    );
    if (flag !== undefined) target.flag = flag;
  };

  const list = pick(root, 'predictions', 'diseases', 'findings', 'results', 'probabilities');
  if (Array.isArray(list) && list.length > 0 && list.every(isRecord)) {
    for (const item of list as UnknownRecord[]) {
      const disease = matchDisease(pick(item, 'disease', 'name', 'label', 'class'));
      if (disease) applyRecord(entry(disease), item);
    }
  } else if (list !== undefined) {
    forEachDiseaseValue(list, (disease, value) => {
      if (isRecord(value)) applyRecord(entry(disease), value);
      else {
        const n = probability(value);
        if (n !== undefined) entry(disease).raw = n;
      }
    });
  }

  forEachDiseaseValue(pick(root, 'raw_probabilities', 'rawProbabilities'), (disease, value) => {
    const n = probability(value);
    if (n !== undefined) entry(disease).raw = n;
  });
  forEachDiseaseValue(
    pick(root, 'calibrated_probabilities', 'calibratedProbabilities'),
    (disease, value) => {
      const n = probability(value);
      if (n !== undefined) entry(disease).cal = n;
    },
  );
  forEachDiseaseValue(pick(root, 'thresholds'), (disease, value) => {
    const n = asNumber(value);
    if (n !== undefined) entry(disease).thr = n;
  });
  forEachDiseaseValue(pick(root, 'predicted_labels', 'binary_predictions'), (disease, value) => {
    const flag = asBoolean(value) ?? (value === 1 ? true : value === 0 ? false : undefined);
    if (flag !== undefined) entry(disease).flag = flag;
  });

  if (ignored > 0) {
    warnings.push(
      `${ignored} probability value${ignored === 1 ? ' was' : 's were'} outside the range 0–1 and ${
        ignored === 1 ? 'was' : 'were'
      } ignored.`,
    );
  }

  return DISEASES.map((disease) => {
    const found = table.get(disease);
    const rawP = found?.raw ?? null;
    const calP = found?.cal ?? null;
    const rawT = found?.rawThr ?? found?.thr ?? null;
    const calT = found?.calThr ?? found?.thr ?? null;
    const rawFlag =
      found?.rawFlag !== undefined
        ? found.rawFlag
        : rawP !== null && rawT !== null
        ? rawP >= rawT
        : found?.flag ?? null;
    const calFlag =
      found?.calFlag !== undefined
        ? found.calFlag
        : calP !== null && calT !== null
        ? calP >= calT
        : found?.flag ?? null;

    return {
      disease,
      rawProbability: rawP,
      calibratedProbability: calP,
      threshold: calT ?? rawT,
      rawThreshold: rawT,
      calibratedThreshold: calT,
      rawFlagged: rawFlag,
      calibratedFlagged: calFlag,
      flagged: calFlag ?? rawFlag,
    };
  });
}

/* ------------------------------------------------------------------ */
/* Metric collections                                                  */
/* ------------------------------------------------------------------ */

const META_KEYS = new Set([
  'status',
  'overallstatus',
  'summary',
  'method',
  'description',
  'notes',
  'metrics',
  'scores',
  'measurements',
  'flags',
  'warnings',
  'perdisease',
  'perclass',
  'bydisease',
  'passed',
  'explanation',
  'interpretation',
  'threshold',
]);

function metricFromEntry(key: string, value: unknown): ReportMetric | undefined {
  if (typeof value === 'number' || typeof value === 'boolean') {
    if (typeof value === 'number' && !Number.isFinite(value)) return undefined;
    return { key, label: humanizeKey(key), value };
  }
  if (typeof value === 'string') {
    const text = value.trim();
    return text !== '' && text.length <= 60 ? { key, label: humanizeKey(key), value: text } : undefined;
  }
  const record = asRecord(value);
  if (!record) return undefined;
  const inner = pick(record, 'value', 'score');
  if (typeof inner !== 'number' && typeof inner !== 'boolean' && typeof inner !== 'string') {
    return undefined;
  }
  if (typeof inner === 'number' && !Number.isFinite(inner)) return undefined;
  return {
    key,
    label: asString(pick(record, 'label', 'title', 'name')) ?? humanizeKey(key),
    value: inner,
    unit: asString(pick(record, 'unit')),
    status: normalizeMetricStatus(pick(record, 'status', 'result')),
    description: asString(pick(record, 'description', 'interpretation', 'explanation')),
  };
}

function metricsFromCollection(collection: unknown): ReportMetric[] {
  const out: ReportMetric[] = [];
  if (Array.isArray(collection)) {
    for (const item of collection) {
      const record = asRecord(item);
      if (!record) continue;
      const key = asString(pick(record, 'key', 'name', 'id', 'metric'));
      if (!key) continue;
      const metric = metricFromEntry(key, record);
      if (metric) out.push(metric);
    }
  } else if (isRecord(collection)) {
    for (const [key, value] of Object.entries(collection)) {
      const metric = metricFromEntry(key, value);
      if (metric) out.push(metric);
    }
  }
  return out;
}

function metricsFrom(record: UnknownRecord): ReportMetric[] {
  const explicit = pick(record, 'metrics', 'scores', 'measurements');
  if (Array.isArray(explicit) || isRecord(explicit)) return metricsFromCollection(explicit);

  const flat: ReportMetric[] = [];
  for (const [key, value] of Object.entries(record)) {
    if (META_KEYS.has(normalizeKey(key))) continue;
    const metric = metricFromEntry(key, value);
    if (metric) flat.push(metric);
  }
  return flat;
}

/* ------------------------------------------------------------------ */
/* Sections                                                            */
/* ------------------------------------------------------------------ */

function normalizeQuality(raw: unknown): QualityAssessment | undefined {
  const record = asRecord(raw);
  if (!record) return undefined;
  const metrics = metricsFrom(record);
  const summary = asString(pick(record, 'summary', 'description', 'notes'));
  const overallStatus = normalizeMetricStatus(pick(record, 'overall_status', 'overallStatus', 'status'));
  if (metrics.length === 0 && !summary && !overallStatus) return undefined;
  return { summary, overallStatus, metrics };
}

function normalizeUncertainty(raw: unknown): UncertaintyAssessment | undefined {
  const record = asRecord(raw);
  if (!record) return undefined;

  const perDisease: Partial<Record<DiseaseName, number>> = {};
  forEachDiseaseValue(pick(record, 'per_disease', 'perDisease', 'per_class', 'by_disease'), (d, v) => {
    const n = asNumber(v);
    if (n !== undefined) perDisease[d] = n;
  });

  const metrics = metricsFrom(record);
  const method = asString(pick(record, 'method'));
  const summary = asString(pick(record, 'summary', 'description', 'notes'));
  if (metrics.length === 0 && Object.keys(perDisease).length === 0 && !summary) return undefined;
  return { method, summary, metrics, perDisease };
}

function normalizeOOD(raw: unknown): OODAssessment | undefined {
  const record = asRecord(raw);
  if (!record) return undefined;
  const status = normalizeOODStatus(pick(record, 'status', 'label', 'result', 'is_ood', 'isOod', 'ood'));
  const score = asNumber(pick(record, 'score', 'ood_score', 'oodScore'));
  const threshold = asNumber(pick(record, 'threshold'));
  const leadDetectorName = asString(pick(record, 'lead_detector_name', 'leadDetectorName'));
  const triggersCount = asNumber(pick(record, 'triggers_count', 'triggersCount'));
  const triggersTotal = asNumber(pick(record, 'triggers_total', 'triggersTotal'));
  const method = asString(pick(record, 'method', 'detector'));
  const explanation = asString(pick(record, 'explanation', 'description', 'summary', 'notes'));

  const rawDetectors = asArray(pick(record, 'detectors', 'detector_results'));
  const detectors: OODDetector[] = [];
  for (const item of rawDetectors) {
    const dRec = asRecord(item);
    if (!dRec) continue;
    const name = asString(pick(dRec, 'name', 'detector', 'method')) ?? 'Detector';
    const s = asNumber(pick(dRec, 'score', 'value'));
    const t = asNumber(pick(dRec, 'threshold'));
    if (s !== undefined && t !== undefined) {
      detectors.push({
        key: asString(pick(dRec, 'key', 'id')),
        name,
        score: s,
        threshold: t,
        status: asString(pick(dRec, 'status')),
        is_ood: asBoolean(pick(dRec, 'is_ood', 'isOod')),
        description: asString(pick(dRec, 'description')),
      });
    }
  }

  if (detectors.length === 0) {
    const dict = asRecord(pick(record, 'method_results'));
    if (dict) {
      for (const [key, val] of Object.entries(dict)) {
        const dRec = asRecord(val);
        if (!dRec) continue;
        const s = asNumber(pick(dRec, 'score'));
        const t = asNumber(pick(dRec, 'threshold'));
        if (s !== undefined && t !== undefined) {
          const name =
            key === 'mahalanobis'
              ? 'Mahalanobis Distance'
              : key === 'energy'
              ? 'Energy Score'
              : key === 'msp'
              ? 'Maximum Sigmoid Probability (MSP)'
              : key;
          detectors.push({
            key,
            name,
            score: s,
            threshold: t,
            status: asString(pick(dRec, 'status')),
            is_ood: asBoolean(pick(dRec, 'is_ood', 'isOod')),
          });
        }
      }
    }
  }

  if (status === undefined && score === undefined && !explanation) return undefined;
  return {
    status,
    score,
    threshold,
    leadDetectorName,
    triggersCount,
    triggersTotal,
    method,
    explanation,
    detectors: detectors.length > 0 ? detectors : undefined,
  };
}

function normalizeCalibration(raw: unknown): CalibrationAssessment | undefined {
  const record = asRecord(raw);
  if (!record) return undefined;

  const bins: ReliabilityBin[] = [];
  const source = pick(record, 'reliability', 'reliability_diagram', 'reliability_bins', 'bins');
  for (const item of asArray(source)) {
    const bin = asRecord(item);
    if (!bin) continue;
    const confidence = asNumber(pick(bin, 'confidence', 'mean_confidence', 'predicted', 'x'));
    const accuracy = asNumber(
      pick(bin, 'accuracy', 'observed', 'observed_frequency', 'fraction_positives', 'y'),
    );
    if (confidence === undefined || accuracy === undefined) continue;
    if (confidence < 0 || confidence > 1 || accuracy < 0 || accuracy > 1) continue;
    bins.push({ confidence, accuracy, count: asNumber(pick(bin, 'count', 'n')) });
  }

  const beforeBins: ReliabilityBin[] = [];
  const beforeSource = pick(record, 'before_bins', 'beforeBins');
  for (const item of asArray(beforeSource)) {
    const bin = asRecord(item);
    if (!bin) continue;
    const confidence = asNumber(pick(bin, 'confidence', 'mean_confidence', 'predicted', 'x'));
    const accuracy = asNumber(
      pick(bin, 'accuracy', 'observed', 'observed_frequency', 'fraction_positives', 'y'),
    );
    if (confidence === undefined || accuracy === undefined) continue;
    if (confidence < 0 || confidence > 1 || accuracy < 0 || accuracy > 1) continue;
    beforeBins.push({ confidence, accuracy, count: asNumber(pick(bin, 'count', 'n')) });
  }

  const compRecord = asRecord(pick(record, 'comparative_metrics', 'comparativeMetrics'));
  let comparativeMetrics: CalibrationAssessment['comparativeMetrics'] = undefined;
  if (compRecord) {
    const parseComp = (k: string) => {
      const it = asRecord(compRecord[k]);
      if (!it) return undefined;
      const before = asNumber(it.before);
      const after = asNumber(it.after);
      if (before === undefined || after === undefined) return undefined;
      const change = asNumber(it.change) ?? after - before;
      const status = asString(it.status) as 'improved' | 'preserved' | 'tradeoff' | 'worsened' | undefined;
      return {
        title: asString(it.title),
        before,
        after,
        change,
        status,
        description: asString(it.description),
      };
    };
    comparativeMetrics = {
      nll: parseComp('nll'),
      brier: parseComp('brier'),
      auroc: parseComp('auroc'),
      ece: parseComp('ece'),
    };
  }

  const result: CalibrationAssessment = {
    method: asString(pick(record, 'method')),
    ece: asNumber(pick(record, 'ece', 'expected_calibration_error')),
    brier: asNumber(pick(record, 'brier', 'brier_score')),
    nll: asNumber(pick(record, 'nll', 'negative_log_likelihood')),
    comparativeMetrics,
    bins,
    beforeBins: beforeBins.length > 0 ? beforeBins : undefined,
    notes: asString(pick(record, 'notes', 'summary', 'description')),
  };
  const hasContent =
    result.method !== undefined ||
    result.ece !== undefined ||
    result.brier !== undefined ||
    result.nll !== undefined ||
    result.notes !== undefined ||
    bins.length > 0;
  return hasContent ? result : undefined;
}

function normalizeExplanationMetric(raw: unknown): ExplanationMetric | undefined {
  const direct = asNumber(raw);
  if (direct !== undefined) return { value: direct };
  const record = asRecord(raw);
  if (!record) return undefined;
  const value = asNumber(pick(record, 'value', 'score'));
  if (value === undefined) return undefined;
  return {
    value,
    metric: asString(pick(record, 'metric', 'name', 'method')),
    description: asString(pick(record, 'description', 'explanation')),
  };
}

function stripExtension(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(0, dot) : name;
}

function normalizeExplanation(raw: unknown): ExplanationAssessment | undefined {
  const record = asRecord(raw);
  if (!record) return undefined;

  const method = asString(pick(record, 'method'));
  const artifacts: ExplanationArtifact[] = [];
  for (const item of asArray(pick(record, 'artifacts', 'images', 'heatmaps'))) {
    if (typeof item === 'string' && item.trim() !== '') {
      artifacts.push({ name: item, label: humanizeKey(stripExtension(item)), method });
      continue;
    }
    const entry = asRecord(item);
    if (!entry) continue;
    const name = asString(pick(entry, 'name', 'artifact_name', 'artifactName', 'id', 'file'));
    if (!name) continue;
    const disease = matchDisease(pick(entry, 'disease', 'class', 'target'));
    const entryMethod = asString(pick(entry, 'method')) ?? method;
    artifacts.push({
      name,
      label:
        asString(pick(entry, 'label', 'title')) ??
        (disease ? `${entryMethod ?? 'Explanation'} — ${disease.replace(/_/g, ' ')}` : humanizeKey(stripExtension(name))),
      method: entryMethod,
      disease,
      caption: asString(pick(entry, 'caption', 'description')),
    });
  }

  const stability = normalizeExplanationMetric(pick(record, 'stability', 'explanation_stability'));
  const faithfulness = normalizeExplanationMetric(
    pick(record, 'faithfulness', 'explanation_faithfulness'),
  );
  if (artifacts.length === 0 && !stability && !faithfulness) return undefined;
  return { method, artifacts, stability, faithfulness };
}

function normalizeSimilarity(raw: unknown): SimilarityAssessment | undefined {
  const record = asRecord(raw);
  if (!record) return undefined;
  const rawMatches = asArray(pick(record, 'matches', 'top_matches', 'results'));
  const matches: SimilarityMatch[] = [];
  rawMatches.forEach((m, idx) => {
    const rec = asRecord(m);
    if (!rec) return;
    const filename = asString(pick(rec, 'filename', 'name', 'file')) ?? `case_${idx + 1}.jpg`;
    const score = asNumber(pick(rec, 'similarity_score', 'score', 'similarity')) ?? 0;
    const rank = asNumber(pick(rec, 'rank')) ?? idx + 1;
    const imageUrl = asString(pick(rec, 'image_url', 'imageUrl', 'url')) ?? `/reference_images/${filename}`;
    const imageId = asString(pick(rec, 'image_id', 'original_image_identifier', 'imageIndex', 'image_index')) ?? filename;
    const metadataRec = asRecord(pick(rec, 'metadata'));
    const provenance = asString(pick(rec, 'provenance', 'dataset_source', 'datasetSource')) ?? (metadataRec ? asString(pick(metadataRec, 'dataset_source', 'datasetSource')) : undefined) ?? 'NIH ChestX-ray14';
    const isSynthetic = asBoolean(pick(rec, 'is_synthetic', 'isSynthetic')) ?? (provenance.toLowerCase().includes('synthetic') || filename.startsWith('xray00') || imageId.startsWith('DEMO-'));

    let findings = asArray(pick(rec, 'findings')).map((s) => String(s));
    if (findings.length === 0 && metadataRec) {
      findings = asArray(pick(metadataRec, 'findings')).map((s) => String(s));
    }
    const findingLabels = asString(pick(rec, 'finding_labels', 'findingLabels')) ?? (metadataRec ? asString(pick(metadataRec, 'finding_labels', 'findingLabels')) : undefined);
    
    const rawConc = asRecord(pick(rec, 'concordance'));
    let concordance: SimilarityMatch['concordance'] = undefined;
    if (rawConc) {
      const cStatus = (asString(pick(rawConc, 'status')) as SimilarityConcordance['status']) || 'NEUTRAL';
      const cLabel = asString(pick(rawConc, 'label')) ?? 'Concordance Evaluated';
      const cDesc = asString(pick(rawConc, 'description')) ?? '';
      const cShared = asArray(pick(rawConc, 'shared_findings')).map((s) => String(s));
      const cRef = asArray(pick(rawConc, 'reference_findings')).map((s) => String(s));
      concordance = {
        status: cStatus,
        label: cLabel,
        description: cDesc,
        matches_primary: asBoolean(pick(rawConc, 'matches_primary')),
        has_overlap: asBoolean(pick(rawConc, 'has_overlap')),
        is_concordant: asBoolean(pick(rawConc, 'is_concordant')),
        shared_findings: cShared,
        reference_findings: cRef,
      };
    }

    matches.push({
      rank,
      filename,
      image_id: imageId,
      original_image_identifier: imageId,
      similarity_score: score,
      image_url: imageUrl,
      findings: findings.length > 0 ? findings : undefined,
      finding_labels: findingLabels,
      provenance,
      is_synthetic: isSynthetic,
      metadata: (metadataRec as SimilarityMatch['metadata']) ?? undefined,
      concordance,
    });
  });

  const topK = asNumber(pick(record, 'top_k', 'topK')) ?? matches.length;
  const totalReferences = asNumber(pick(record, 'total_references', 'totalReferences')) ?? matches.length;
  const disclaimer = asString(pick(record, 'disclaimer')) ?? 'Similarity scores represent image/feature similarity only. They are not a medical diagnosis or clinical recommendation.';
  const explanationText = asString(pick(record, 'explanation_text', 'explanationText')) ?? 'NIH Dataset Prevalence measures label frequency across the dataset. Retrieved-Case Label Support measures label representation among visually similar references. Neither is a calibrated disease probability.';
  const summary = asString(pick(record, 'summary', 'description')) ?? `Retrieved Top-${matches.length} similar reference radiographs.`;
  const status = asString(pick(record, 'status')) ?? 'SUCCESS';

  const concordanceRate = asNumber(pick(record, 'concordance_rate', 'concordanceRate'));
  const concordantCasesCount = asNumber(pick(record, 'concordant_cases_count', 'concordantCasesCount'));
  const primaryConcordanceRate = asNumber(pick(record, 'primary_concordance_rate', 'primaryConcordanceRate'));
  const primaryConcordantCount = asNumber(pick(record, 'primary_concordant_count', 'primaryConcordantCount'));
  const anyFindingOverlapRate = asNumber(pick(record, 'any_finding_overlap_rate', 'anyFindingOverlapRate'));
  const anyFindingOverlapCount = asNumber(pick(record, 'any_finding_overlap_count', 'anyFindingOverlapCount'));
  const primaryQueryFinding = asString(pick(record, 'primary_query_finding', 'primaryQueryFinding'));
  const concordanceSummary = asString(pick(record, 'concordance_summary', 'concordanceSummary'));

  const rawGlobalStats = asArray(pick(record, 'global_dataset_statistics', 'globalDatasetStatistics'));
  const globalDatasetStatistics: GlobalDiseaseStatistic[] = rawGlobalStats.map((item) => {
    const gRec = asRecord(item) ?? {};
    return {
      disease_name: asString(pick(gRec, 'disease_name', 'diseaseName', 'disease')) ?? '',
      occurrence_count: asNumber(pick(gRec, 'occurrence_count', 'occurrenceCount', 'count')) ?? 0,
      prevalence_percentage: asNumber(pick(gRec, 'prevalence_percentage', 'prevalencePercentage', 'prevalence')) ?? 0,
      denominator: asNumber(pick(gRec, 'denominator', 'total')) ?? 112120,
    };
  }).filter((g) => g.disease_name);

  const rawNoFinding = asRecord(pick(record, 'no_finding_statistics', 'noFindingStatistics', 'no_finding'));
  const noFindingStatistics: NoFindingStatistic | undefined = rawNoFinding ? {
    disease_name: asString(pick(rawNoFinding, 'disease_name', 'diseaseName', 'label')) ?? 'No Finding',
    occurrence_count: asNumber(pick(rawNoFinding, 'occurrence_count', 'occurrenceCount', 'count')) ?? 0,
    prevalence_percentage: asNumber(pick(rawNoFinding, 'prevalence_percentage', 'prevalencePercentage', 'prevalence')) ?? 0,
    denominator: asNumber(pick(rawNoFinding, 'denominator', 'total')) ?? 112120,
  } : undefined;

  const rawSupport = asArray(pick(record, 'weighted_label_support', 'weightedLabelSupport', 'retrieved_case_support', 'retrievedCaseSupport'));
  const weightedLabelSupport = rawSupport.map((item) => {
    const sRec = asRecord(item) ?? {};
    const diseaseName = asString(pick(sRec, 'disease', 'disease_name', 'diseaseName')) ?? '';
    const rawWeighted = asNumber(pick(sRec, 'weighted_support', 'weightedSupport'));
    const isAvail = asBoolean(pick(sRec, 'is_available', 'isAvailable'));
    const reason = asString(pick(sRec, 'reason'));

    return {
      disease: diseaseName,
      disease_name: diseaseName,
      weighted_support: rawWeighted !== undefined ? rawWeighted : (isAvail === false ? null : 0),
      retrieved_count: asNumber(pick(sRec, 'retrieved_count', 'retrievedCount')) ?? 0,
      retrieval_rate: asNumber(pick(sRec, 'retrieval_rate', 'retrievalRate')) ?? 0,
      retrieved_denominator: asNumber(pick(sRec, 'retrieved_denominator', 'retrievedDenominator')) ?? topK,
      dataset_prevalence: asNumber(pick(sRec, 'dataset_prevalence', 'datasetPrevalence')),
      dataset_occurrence_count: asNumber(pick(sRec, 'dataset_occurrence_count', 'datasetOccurrenceCount')),
      dataset_denominator: asNumber(pick(sRec, 'dataset_denominator', 'datasetDenominator')),
      is_available: isAvail !== undefined ? isAvail : (rawWeighted !== undefined),
      reason: reason || undefined,
      top_k: asNumber(pick(sRec, 'top_k', 'topK')) ?? topK,
    };
  }).filter((s) => s.disease);

  const rawPredictedEvidence = asArray(pick(record, 'predicted_findings_evidence', 'predictedFindingsEvidence', 'predicted_evidence', 'predictedEvidence'));
  const predictedFindingsEvidence: PredictedFindingEvidence[] = rawPredictedEvidence.map((item) => {
    const pRec = asRecord(item) ?? {};
    const predFinding = asString(pick(pRec, 'predicted_finding', 'predictedFinding', 'finding', 'disease')) ?? '';
    const matchingCases = asNumber(pick(pRec, 'matching_cases', 'matchingCases')) ?? 0;
    const retrievedCases = asNumber(pick(pRec, 'retrieved_cases', 'retrievedCases')) ?? topK;
    const matchingRate = asNumber(pick(pRec, 'matching_rate', 'matchingRate')) ?? 0;
    const weightedSupport = asNumber(pick(pRec, 'weighted_support', 'weightedSupport')) ?? 0;
    const datasetPrevalence = asNumber(pick(pRec, 'dataset_prevalence', 'datasetPrevalence'));
    const isMatched = asBoolean(pick(pRec, 'is_matched', 'isMatched')) ?? (matchingCases > 0);
    const explanation = asString(pick(pRec, 'explanation'));
    const matchingRefs: SimilarityMatch[] = asArray(pick(pRec, 'matching_references', 'matchingReferences')).map((r) => {
      const rRec = asRecord(r) ?? {};
      return {
        rank: asNumber(pick(rRec, 'rank')) ?? 0,
        filename: asString(pick(rRec, 'filename', 'image_id')) ?? '',
        image_id: asString(pick(rRec, 'image_id', 'filename')),
        similarity_score: asNumber(pick(rRec, 'similarity_score', 'similarityScore')) ?? 0,
        image_url: asString(pick(rRec, 'image_url', 'imageUrl')) ?? '',
        findings: asArray(pick(rRec, 'findings')).map(asString).filter(Boolean) as string[],
        finding_labels: asString(pick(rRec, 'finding_labels', 'findingLabels')),
        provenance: asString(pick(rRec, 'provenance', 'dataset_source')),
        is_synthetic: asBoolean(pick(rRec, 'is_synthetic', 'isSynthetic')),
        metadata: asRecord(pick(rRec, 'metadata')) as SimilarityMatch['metadata'],
      };
    });

    return {
      predicted_finding: predFinding,
      matching_cases: matchingCases,
      retrieved_cases: retrievedCases,
      matching_rate: matchingRate,
      weighted_support: weightedSupport,
      dataset_prevalence: datasetPrevalence,
      is_matched: isMatched,
      matching_references: matchingRefs,
      explanation: explanation || undefined,
    };
  }).filter((p) => p.predicted_finding);

  const rawPredictedFindings = asArray(pick(record, 'predicted_findings', 'predictedFindings')).map(asString).filter(Boolean) as string[];

  if (matches.length === 0 && !summary) return undefined;

  return {
    status,
    top_k: topK,
    requested_k: asNumber(pick(record, 'requested_k', 'requestedK')) ?? topK,
    total_references: totalReferences,
    candidates_searched: asNumber(pick(record, 'candidates_searched', 'candidatesSearched')) ?? totalReferences,
    matches,
    top_match: matches[0] ?? null,
    concordance_rate: concordanceRate,
    concordant_cases_count: concordantCasesCount,
    primary_concordance_rate: primaryConcordanceRate,
    primary_concordant_count: primaryConcordantCount,
    any_finding_overlap_rate: anyFindingOverlapRate,
    any_finding_overlap_count: anyFindingOverlapCount,
    primary_query_finding: primaryQueryFinding,
    concordance_summary: concordanceSummary,
    predicted_findings_evidence: predictedFindingsEvidence.length > 0 ? predictedFindingsEvidence : undefined,
    predicted_evidence: predictedFindingsEvidence.length > 0 ? predictedFindingsEvidence : undefined,
    predicted_findings: rawPredictedFindings.length > 0 ? rawPredictedFindings : undefined,
    global_dataset_statistics: globalDatasetStatistics.length > 0 ? globalDatasetStatistics : undefined,
    no_finding_statistics: noFindingStatistics,
    weighted_label_support: weightedLabelSupport.length > 0 ? weightedLabelSupport : undefined,
    retrieved_case_support: weightedLabelSupport.length > 0 ? weightedLabelSupport : undefined,
    explanation_text: explanationText,
    disclaimer,
    summary,
  };
}

const IMAGE_KNOWN_KEYS = new Set([
  'filename',
  'name',
  'width',
  'height',
  'format',
  'mimetype',
  'contenttype',
  'sizebytes',
  'size',
  'artifactname',
  'artifact',
  'originalartifact',
  'url',
  'path',
]);

function normalizeImage(root: UnknownRecord): { info: ImageInfo; filename?: string } {
  const record = asRecord(pick(root, 'image', 'input_image')) ?? {};
  const filename =
    asString(pick(record, 'filename', 'name', 'original_filename')) ??
    asString(pick(root, 'filename', 'original_filename', 'file_name', 'image_filename'));

  const extra: Array<{ label: string; value: string }> = [];
  const sources: UnknownRecord[] = [record];
  const metadata = asRecord(pick(root, 'metadata', 'image_metadata'));
  if (metadata) sources.push(metadata);
  for (const source of sources) {
    for (const [key, value] of Object.entries(source)) {
      if (source === record && IMAGE_KNOWN_KEYS.has(normalizeKey(key))) continue;
      if (typeof value === 'string' && value.trim() !== '' && value.length <= 80) {
        extra.push({ label: humanizeKey(key), value });
      } else if (typeof value === 'number' && Number.isFinite(value)) {
        extra.push({ label: humanizeKey(key), value: String(value) });
      } else if (typeof value === 'boolean') {
        extra.push({ label: humanizeKey(key), value: value ? 'Yes' : 'No' });
      }
    }
  }

  return {
    filename,
    info: {
      artifactName: asString(
        pick(record, 'artifact_name', 'artifactName', 'artifact', 'original_artifact'),
      ),
      width: asNumber(pick(record, 'width')),
      height: asNumber(pick(record, 'height')),
      format: asString(pick(record, 'format', 'mime_type', 'content_type')),
      sizeBytes: asNumber(pick(record, 'size_bytes', 'sizeBytes', 'size')),
      extra,
    },
  };
}

/* ------------------------------------------------------------------ */
/* Reports and summaries                                               */
/* ------------------------------------------------------------------ */

export function normalizeReport(raw: unknown): AnalysisReport {
  const root = unwrapEnvelope(raw);
  if (!root) throw badResponse();
  const id = asString(pick(root, 'id', 'analysis_id', 'analysisId'));
  if (!id) throw badResponse();

  const warnings: string[] = asArray(pick(root, 'warnings'))
    .map((w) => asString(w))
    .filter((w): w is string => w !== undefined);

  const { info, filename } = normalizeImage(root);
  const modelRecord = asRecord(pick(root, 'model'));

  return {
    id,
    createdAt: asString(pick(root, 'created_at', 'createdAt', 'timestamp', 'completed_at')),
    filename,
    status: normalizeStatus(pick(root, 'status', 'state')),
    error: asString(pick(root, 'error', 'error_message', 'failure_reason')),
    image: info,
    predictions: normalizePredictions(root, warnings),
    quality: normalizeQuality(pick(root, 'image_quality', 'quality', 'quality_assessment')),
    uncertainty: normalizeUncertainty(pick(root, 'uncertainty', 'uncertainty_assessment')),
    ood: normalizeOOD(pick(root, 'ood', 'out_of_distribution', 'ood_assessment')),
    calibration: normalizeCalibration(pick(root, 'calibration', 'calibration_metrics')),
    explanation: normalizeExplanation(pick(root, 'explanation', 'explainability', 'xai')),
    similarity: normalizeSimilarity(pick(root, 'similarity', 'similarity_analysis', 'similar_cases')),
    model: modelRecord
      ? {
          name: asString(pick(modelRecord, 'name')),
          version: asString(pick(modelRecord, 'version')),
        }
      : undefined,
    warnings,
    isDemo: false,
  };
}

export function normalizeSummary(raw: unknown): AnalysisSummary | null {
  const root = unwrapEnvelope(raw);
  if (!root) return null;
  const id = asString(pick(root, 'id', 'analysis_id', 'analysisId'));
  if (!id) return null;

  const { filename } = normalizeImage(root);

  let topFinding: AnalysisSummary['topFinding'];
  const declared = asRecord(pick(root, 'top_finding', 'topFinding'));
  const declaredDisease = declared ? matchDisease(pick(declared, 'disease', 'name', 'label')) : undefined;
  const declaredProbability = declared ? asNumber(pick(declared, 'probability', 'score')) : undefined;
  if (declaredDisease && declaredProbability !== undefined && declaredProbability >= 0 && declaredProbability <= 1) {
    topFinding = {
      disease: declaredDisease,
      probability: declaredProbability,
      calibrated: asBoolean(pick(declared ?? {}, 'calibrated')) ?? false,
    };
  } else if (pick(root, 'predictions', 'raw_probabilities', 'calibrated_probabilities') !== undefined) {
    const predictions = normalizePredictions(root, []);
    const best = topPredictions({ predictions }, 1)[0];
    if (best) {
      topFinding = {
        disease: best.prediction.disease,
        probability: best.value,
        calibrated: best.prediction.calibratedProbability !== null,
      };
    }
  }

  const oodRecord = asRecord(pick(root, 'ood', 'out_of_distribution'));
  const qualityRecord = asRecord(pick(root, 'image_quality', 'quality'));

  return {
    id,
    createdAt: asString(pick(root, 'created_at', 'createdAt', 'timestamp')),
    filename,
    status: normalizeStatus(pick(root, 'status', 'state')),
    topFinding,
    oodStatus: normalizeOODStatus(pick(root, 'ood_status') ?? pick(oodRecord ?? {}, 'status', 'is_ood')),
    qualityStatus: normalizeMetricStatus(
      pick(root, 'quality_status') ?? pick(qualityRecord ?? {}, 'overall_status', 'status'),
    ),
    isDemo: false,
  };
}

export function normalizeList(
  raw: unknown,
  requested: { page: number; pageSize: number },
): Paginated<AnalysisSummary> {
  const root = asRecord(raw);
  const rawItems = Array.isArray(raw)
    ? raw
    : asArray(pick(root, 'items', 'results', 'analyses', 'data', 'records'));
  if (!Array.isArray(raw) && !root) throw badResponse();

  const items = rawItems
    .map((item) => normalizeSummary(item))
    .filter((item): item is AnalysisSummary => item !== null);

  const total = asNumber(pick(root, 'total', 'count', 'total_count'));
  const page = asNumber(pick(root, 'page')) ?? requested.page;
  const pageSize = asNumber(pick(root, 'page_size', 'pageSize', 'limit')) ?? requested.pageSize;

  const explicitMore = pick(root, 'has_more', 'hasMore');
  const hasMore =
    typeof explicitMore === 'boolean'
      ? explicitMore
      : total !== undefined
        ? page * pageSize < total
        : rawItems.length >= pageSize;

  return { items, total: total ?? null, page, pageSize, hasMore };
}

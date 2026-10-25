/**
 * Normalised analysis types used throughout the UI.
 *
 * Optional sections are optional on purpose: the interface must never invent
 * values. Anything the backend did not return is `undefined`/`null` and is
 * rendered as "not evaluated" or "unavailable".
 */

/** The 14 findings, in the canonical order used by every list and table. */
export const DISEASES = [
  'Atelectasis',
  'Cardiomegaly',
  'Effusion',
  'Infiltration',
  'Mass',
  'Nodule',
  'Pneumonia',
  'Pneumothorax',
  'Consolidation',
  'Edema',
  'Emphysema',
  'Fibrosis',
  'Pleural_Thickening',
  'Hernia',
] as const;

export type DiseaseName = (typeof DISEASES)[number];

export function formatDiseaseName(name: DiseaseName): string {
  return name.replace(/_/g, ' ');
}

export type AnalysisStatus = 'queued' | 'processing' | 'completed' | 'failed' | 'unknown';

export interface DiseasePrediction {
  disease: DiseaseName;
  /** Uncalibrated model output in [0, 1], or null if not supplied. */
  rawProbability: number | null;
  /** Calibrated probability in [0, 1], or null if not supplied. */
  calibratedProbability: number | null;
  /** Decision threshold, only when the backend provides one. */
  threshold: number | null;
  /** Uncalibrated model decision threshold in [0, 1]. */
  rawThreshold?: number | null;
  /** Temperature-scaled equivalent decision threshold in [0, 1]. */
  calibratedThreshold?: number | null;
  /** Whether the backend flagged the finding as positive in raw probability space. */
  rawFlagged?: boolean | null;
  /** Whether the backend flagged the finding as positive in calibrated probability space. */
  calibratedFlagged?: boolean | null;
  /** Whether the backend flagged the finding as positive, if it says so. */
  flagged: boolean | null;
}

export type MetricStatus = 'good' | 'caution' | 'poor' | 'info';

export interface ReportMetric {
  key: string;
  label: string;
  value: number | string | boolean;
  unit?: string;
  status?: MetricStatus;
  /** Backend-supplied explanation, if any. */
  description?: string;
}

export interface QualityAssessment {
  summary?: string;
  overallStatus?: MetricStatus;
  metrics: ReportMetric[];
}

export interface UncertaintyAssessment {
  method?: string;
  summary?: string;
  metrics: ReportMetric[];
  perDisease: Partial<Record<DiseaseName, number>>;
}

export type OODStatus = 'in_distribution' | 'out_of_distribution' | 'uncertain';

export interface OODDetector {
  key?: string;
  name: string;
  score: number;
  threshold: number;
  status?: string;
  is_ood?: boolean;
  description?: string;
}

export interface OODAssessment {
  status?: OODStatus;
  score?: number;
  threshold?: number;
  leadDetectorName?: string;
  triggersCount?: number;
  triggersTotal?: number;
  method?: string;
  explanation?: string;
  detectors?: OODDetector[];
}

export interface ReliabilityBin {
  /** Mean predicted confidence in the bin. */
  confidence: number;
  /** Observed frequency in the bin. */
  accuracy: number;
  count?: number;
}

export interface ComparativeCalibrationMetric {
  title?: string;
  before: number;
  after: number;
  change: number;
  status?: 'improved' | 'preserved' | 'tradeoff' | 'worsened';
  description?: string;
}

export interface CalibrationAssessment {
  method?: string;
  ece?: number;
  brier?: number;
  nll?: number;
  comparativeMetrics?: {
    nll?: ComparativeCalibrationMetric;
    brier?: ComparativeCalibrationMetric;
    auroc?: ComparativeCalibrationMetric;
    ece?: ComparativeCalibrationMetric;
  };
  bins: ReliabilityBin[];
  beforeBins?: ReliabilityBin[];
  notes?: string;
}

export interface ExplanationArtifact {
  /** Artifact name used with GET /analyses/:id/artifacts/:artifactName. */
  name: string;
  label: string;
  method?: string;
  disease?: DiseaseName;
  caption?: string;
}

export interface ExplanationMetric {
  value: number | string;
  metric?: string;
  description?: string;
}

export interface ExplanationAssessment {
  method?: string;
  spatialReceptiveField?: string;
  interpretationDisclaimer?: string;
  artifacts: ExplanationArtifact[];
  stability?: ExplanationMetric;
  faithfulness?: ExplanationMetric;
}

export interface ImageInfo {
  /** Artifact name of the original image (defaults to "original"). */
  artifactName?: string;
  width?: number;
  height?: number;
  format?: string;
  sizeBytes?: number;
  /** Any further primitive metadata returned by the API. */
  extra: Array<{ label: string; value: string }>;
}

export interface SimilarityMetadata {
  finding_labels?: string;
  findings?: string[];
  patient_age?: number | string | null;
  patient_gender?: string;
  view_position?: string;
  clinical_notes?: string;
}

export interface SimilarityConcordance {
  status: 'PRIMARY_CONCORDANT' | 'SECONDARY_OVERLAP' | 'CONCORDANT' | 'DIFFERENTIAL' | 'NO_FINDING' | 'NEUTRAL';
  label: string;
  matches_primary?: boolean;
  has_overlap?: boolean;
  is_concordant?: boolean;
  shared_findings?: string[];
  reference_findings?: string[];
  description?: string;
}

export interface GlobalDiseaseStatistic {
  disease_name: string;
  occurrence_count: number;
  prevalence_percentage: number;
  denominator: number;
}

export interface NoFindingStatistic {
  disease_name: string;
  occurrence_count: number;
  prevalence_percentage: number;
  denominator: number;
}

export interface DiseaseLabelSupport {
  disease: string;
  disease_name?: string;
  weighted_support: number | null;
  retrieved_count: number;
  retrieval_rate: number;
  retrieved_denominator?: number;
  dataset_prevalence?: number | null;
  dataset_occurrence_count?: number | null;
  dataset_denominator?: number | null;
  is_available?: boolean;
  reason?: string | null;
  top_k: number;
}

export interface SimilarityMatch {
  rank: number;
  filename: string;
  image_id?: string;
  original_image_identifier?: string;
  similarity_score: number;
  image_url: string;
  findings?: string[];
  finding_labels?: string;
  provenance?: string;
  is_synthetic?: boolean;
  metadata?: SimilarityMetadata & Record<string, unknown>;
  concordance?: SimilarityConcordance;
}

export interface PredictedFindingEvidence {
  predicted_finding: string;
  matching_cases: number;
  retrieved_cases: number;
  matching_rate: number;
  weighted_support: number;
  dataset_prevalence?: number;
  is_matched: boolean;
  matching_references?: SimilarityMatch[];
  explanation?: string;
}

export interface SimilarityAssessment {
  status: string;
  top_k: number;
  requested_k?: number;
  total_references: number;
  candidates_searched?: number;
  matches: SimilarityMatch[];
  top_match?: SimilarityMatch | null;
  concordance_rate?: number;
  concordant_cases_count?: number;
  primary_concordance_rate?: number;
  primary_concordant_count?: number;
  any_finding_overlap_rate?: number;
  any_finding_overlap_count?: number;
  primary_query_finding?: string;
  concordance_summary?: string;
  predicted_findings_evidence?: PredictedFindingEvidence[];
  predicted_evidence?: PredictedFindingEvidence[];
  predicted_findings?: string[];
  global_dataset_statistics?: GlobalDiseaseStatistic[];
  no_finding_statistics?: NoFindingStatistic;
  weighted_label_support?: DiseaseLabelSupport[];
  retrieved_case_support?: DiseaseLabelSupport[];
  explanation_text?: string;
  disclaimer: string;
  summary: string;
}

export interface ReferenceReport {
  schema_version: string;
  reference_id: string;
  dataset_provenance: string;
  dataset_split?: string;
  patient_id?: string;
  demographics?: {
    patient_age?: number;
    patient_gender?: string;
    view_position?: string;
  };
  dataset_annotations?: {
    findings?: string[];
    finding_labels?: string;
    annotation_type?: string;
    is_authentic_nih?: boolean;
  };
  predictions?: {
    raw_scores?: Record<string, number>;
    calibrated_scores?: Record<string, number>;
    decisions?: Record<string, boolean>;
    positive_findings?: string[];
  };
  image_quality?: {
    composite_score?: number;
    sharpness?: number;
    contrast?: number;
    brightness?: number;
    suitable?: boolean;
    status?: string;
  };
  ood?: {
    is_ood?: boolean;
    status?: string;
    msp_score?: number;
    energy_score?: number;
  };
}

export interface AnalysisReport {
  id: string;
  createdAt?: string;
  filename?: string;
  status: AnalysisStatus;
  /** Failure message when status is "failed". */
  error?: string;
  image: ImageInfo;
  /** Always the 14 canonical diseases, in canonical order. */
  predictions: DiseasePrediction[];
  quality?: QualityAssessment;
  uncertainty?: UncertaintyAssessment;
  ood?: OODAssessment;
  calibration?: CalibrationAssessment;
  explanation?: ExplanationAssessment;
  similarity?: SimilarityAssessment;
  model?: { name?: string; version?: string };
  /** Non-fatal notes, e.g. values ignored while parsing. */
  warnings: string[];
  /** True only for isolated sample data created by demo mode. */
  isDemo: boolean;
}

export interface AnalysisSummary {
  id: string;
  createdAt?: string;
  filename?: string;
  status: AnalysisStatus;
  topFinding?: { disease: DiseaseName; probability: number; calibrated: boolean };
  oodStatus?: OODStatus;
  qualityStatus?: MetricStatus;
  isDemo: boolean;
}

/** Calibrated probability if present, otherwise raw, otherwise null. */
export function primaryProbability(prediction: DiseasePrediction): number | null {
  return prediction.calibratedProbability ?? prediction.rawProbability;
}

export function hasPredictions(report: Pick<AnalysisReport, 'predictions'>): boolean {
  return report.predictions.some(
    (p) => p.rawProbability !== null || p.calibratedProbability !== null,
  );
}

export function topPredictions(report: Pick<AnalysisReport, 'predictions'>, count: number) {
  return report.predictions
    .map((p) => ({ prediction: p, value: primaryProbability(p) }))
    .filter((entry): entry is { prediction: DiseasePrediction; value: number } => entry.value !== null)
    .sort((a, b) => b.value - a.value)
    .slice(0, count);
}

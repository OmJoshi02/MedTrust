import {
  DISEASES,
  formatDiseaseName,
  topPredictions,
  type AnalysisReport,
  type AnalysisSummary,
  type DiseaseName,
  type DiseasePrediction,
  type MetricStatus,
  type ReliabilityBin,
} from '@/types/analysis';
import { clamp, mulberry32, shuffle } from './seed';

/**
 * Clearly labelled SAMPLE data for demo mode. None of these values were
 * produced by a model. They exist only to preview layouts, charts and states.
 */

export type SampleVariant = 'full' | 'minimal' | 'ood';

export interface SampleOptions {
  id: string;
  filename: string;
  createdAt: string;
  seed: number;
  variant?: SampleVariant;
}

const logit = (p: number) => Math.log(clamp(p, 1e-4, 1 - 1e-4) / (1 - clamp(p, 1e-4, 1 - 1e-4)));
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const round = (value: number, digits = 3) => Number(value.toFixed(digits));

function statusFor(value: number, goodAbove: number, cautionAbove: number): MetricStatus {
  if (value >= goodAbove) return 'good';
  if (value >= cautionAbove) return 'caution';
  return 'poor';
}

export function buildSampleReport(options: SampleOptions): AnalysisReport {
  const { id, filename, createdAt, seed, variant = 'full' } = options;
  const random = mulberry32(seed);

  // One or two elevated findings, the rest low.
  const elevated = new Set<DiseaseName>(shuffle(DISEASES, random).slice(0, 1 + Math.floor(random() * 2)));
  const raw = DISEASES.map((d) => (elevated.has(d) ? 0.46 + random() * 0.38 : 0.02 + random() * 0.17));
  const temperature = 1.45;
  const calibrated = raw.map((p) => sigmoid(logit(p) / temperature));

  const minimal = variant === 'minimal';
  const predictions: DiseasePrediction[] = DISEASES.map((disease, index) => {
    const calibratedValue = round(calibrated[index] ?? 0);
    return {
      disease,
      rawProbability: round(raw[index] ?? 0),
      calibratedProbability: minimal ? null : calibratedValue,
      threshold: minimal ? null : 0.5,
      flagged: minimal ? null : calibratedValue >= 0.5,
    };
  });

  const base: AnalysisReport = {
    id,
    createdAt,
    filename,
    status: 'completed',
    image: {
      artifactName: 'original',
      width: 1024,
      height: 1024,
      format: filename.toLowerCase().endsWith('.png') ? 'PNG' : 'JPEG',
      extra: [{ label: 'Source', value: 'Sample metadata (demo mode)' }],
    },
    predictions,
    warnings: [],
    model: { name: 'Demo sample generator' },
    isDemo: true,
  };

  // Image quality
  const sharpness = round(0.62 + random() * 0.3, 2);
  const contrast = round(0.55 + random() * 0.35, 2);
  const brightness = round(0.42 + random() * 0.25, 2);
  const noise = round(0.04 + random() * 0.14, 2);

  if (minimal) {
    // Deliberately sparse: shows how the interface handles missing sections.
    return {
      ...base,
      quality: {
        metrics: [
          { key: 'sharpness', label: 'Sharpness', value: sharpness, status: statusFor(sharpness, 0.7, 0.5) },
          { key: 'contrast', label: 'Contrast', value: contrast, status: statusFor(contrast, 0.6, 0.4) },
        ],
      },
    };
  }

  const ood = variant === 'ood';
  const qualityStatus: MetricStatus = ood ? 'caution' : 'good';

  const perDisease: Partial<Record<DiseaseName, number>> = {};
  DISEASES.forEach((disease, index) => {
    const p = calibrated[index] ?? 0;
    perDisease[disease] = round(clamp(p * (1 - p) * 1.6 + random() * 0.03, 0, 1));
  });
  const perValues = Object.values(perDisease) as number[];
  const meanUncertainty = round(perValues.reduce((a, b) => a + b, 0) / perValues.length);

  const bins: ReliabilityBin[] = Array.from({ length: 10 }, (_, i) => {
    const confidence = round(0.05 + i * 0.1, 2);
    return {
      confidence,
      accuracy: round(clamp(confidence + (random() - 0.5) * 0.12, 0, 1), 2),
      count: Math.round(40 + random() * 160),
    };
  });

  const ranked = topPredictions({ predictions }, 2);
  const artifacts = ranked.map(({ prediction }) => ({
    name: `gradcam_${prediction.disease.toLowerCase()}`,
    label: `Grad-CAM — ${formatDiseaseName(prediction.disease)}`,
    method: 'Grad-CAM (sample)',
    disease: prediction.disease,
    caption: `Sample illustration of a Grad-CAM overlay for ${formatDiseaseName(prediction.disease)}.`,
  }));

  return {
    ...base,
    quality: {
      overallStatus: qualityStatus,
      summary: ood
        ? 'Sample data: image quality is borderline, with reduced sharpness.'
        : 'Sample data: image quality is adequate for analysis.',
      metrics: [
        { key: 'sharpness', label: 'Sharpness', value: ood ? 0.41 : sharpness, status: ood ? 'caution' : statusFor(sharpness, 0.7, 0.5) },
        { key: 'contrast', label: 'Contrast', value: contrast, status: statusFor(contrast, 0.6, 0.4) },
        { key: 'brightness', label: 'Brightness', value: brightness, status: 'good' },
        { key: 'noise', label: 'Noise', value: noise, status: noise < 0.12 ? 'good' : 'caution' },
        { key: 'resolution', label: 'Resolution', value: '1024 × 1024 px', status: 'info' },
      ],
    },
    uncertainty: {
      method: 'Monte Carlo sampling (sample values)',
      summary: 'Sample values for previewing the layout. Higher values mean the model output is less settled.',
      metrics: [
        { key: 'mean_uncertainty', label: 'Mean uncertainty', value: meanUncertainty },
        { key: 'predictive_entropy', label: 'Predictive entropy', value: round(0.18 + random() * 0.3, 3) },
        { key: 'mutual_information', label: 'Mutual information', value: round(0.02 + random() * 0.08, 3) },
      ],
      perDisease,
    },
    ood: ood
      ? {
          status: 'out_of_distribution',
          score: 0.81,
          threshold: 0.62,
          method: 'Feature-space distance (sample)',
          explanation:
            'Sample data: this image is shown as unlike the data the model was developed on, so its predictions should be treated with extra caution.',
        }
      : {
          status: 'in_distribution',
          score: round(0.18 + random() * 0.25, 2),
          threshold: 0.62,
          method: 'Feature-space distance (sample)',
          explanation:
            'Sample data: the image resembles the data the model was developed on. This does not make a prediction correct.',
        },
    calibration: {
      method: 'Temperature scaling (sample)',
      ece: round(0.02 + random() * 0.03, 3),
      brier: round(0.05 + random() * 0.04, 3),
      nll: round(0.16 + random() * 0.08, 3),
      bins,
      notes: 'Sample calibration values for previewing the layout.',
    },
    explanation: {
      method: 'Grad-CAM (sample)',
      artifacts,
      stability: { value: round(0.7 + random() * 0.2, 2), metric: 'Rank correlation (sample)' },
      faithfulness: { value: round(0.55 + random() * 0.25, 2), metric: 'Deletion AUC (sample)' },
    },
    similarity: {
      status: 'SUCCESS',
      top_k: 5,
      total_references: 6,
      matches: [
        { rank: 1, filename: 'xray002.jpg', similarity_score: 84.5, image_url: '/reference_images/xray002.jpg' },
        { rank: 2, filename: 'xray004.jpg', similarity_score: 81.2, image_url: '/reference_images/xray004.jpg' },
        { rank: 3, filename: 'sample_query.jpg', similarity_score: 79.8, image_url: '/reference_images/sample_query.jpg' },
        { rank: 4, filename: 'xray005.jpg', similarity_score: 76.4, image_url: '/reference_images/xray005.jpg' },
        { rank: 5, filename: 'xray003.jpg', similarity_score: 73.1, image_url: '/reference_images/xray003.jpg' },
      ],
      disclaimer:
        'Similarity scores represent image/feature similarity only. They are not a medical diagnosis or clinical recommendation.',
      summary:
        'Sample data: 5 nearest matching reference radiographs based on DenseNet-121 feature embeddings.',
    },
  };
}

const DAY = 86_400_000;
const daysAgo = (n: number, hour = 10) => {
  const date = new Date(Date.now() - n * DAY);
  date.setHours(hour, 12, 0, 0);
  return date.toISOString();
};

/** Pre-existing sample records shown in demo-mode history. */
export const SAMPLE_REPORTS: AnalysisReport[] = [
  buildSampleReport({ id: 'demo-sample-001', filename: 'sample_pa_view_01.png', createdAt: daysAgo(1), seed: 1101 }),
  buildSampleReport({ id: 'demo-sample-002', filename: 'sample_pa_view_02.png', createdAt: daysAgo(3, 15), seed: 2202 }),
  buildSampleReport({ id: 'demo-sample-003', filename: 'sample_lateral_blurred.jpg', createdAt: daysAgo(6), seed: 3303, variant: 'ood' }),
  buildSampleReport({ id: 'demo-sample-004', filename: 'sample_incomplete_result.png', createdAt: daysAgo(9, 9), seed: 4404, variant: 'minimal' }),
  buildSampleReport({ id: 'demo-sample-005', filename: 'sample_pa_view_03.webp', createdAt: daysAgo(14, 13), seed: 5505 }),
  buildSampleReport({ id: 'demo-sample-006', filename: 'sample_pa_view_04.png', createdAt: daysAgo(21), seed: 6606 }),
];

export function summarizeReport(report: AnalysisReport): AnalysisSummary {
  const best = topPredictions(report, 1)[0];
  return {
    id: report.id,
    createdAt: report.createdAt,
    filename: report.filename,
    status: report.status,
    topFinding: best
      ? {
          disease: best.prediction.disease,
          probability: best.value,
          calibrated: best.prediction.calibratedProbability !== null,
        }
      : undefined,
    oodStatus: report.ood?.status,
    qualityStatus: report.quality?.overallStatus,
    isDemo: true,
  };
}

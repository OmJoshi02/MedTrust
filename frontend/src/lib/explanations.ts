import { normalizeKey } from './safe';

/**
 * Plain-language explanations for metrics the backend may return.
 * These describe what a metric means in general. They never describe a
 * specific result, and are shown only next to a value the API supplied.
 */

const QUALITY: Record<string, string> = {
  sharpness: 'How crisp edges and fine structures appear. Low sharpness can hide small findings.',
  blur: 'Estimated amount of blur. Higher values mean less detail is recoverable.',
  laplacianvariance: 'A common sharpness proxy. Lower values suggest a blurrier image.',
  contrast: 'Separation between dark and bright regions. Low contrast can mask subtle opacities.',
  brightness: 'Average intensity of the image. Very dark or very bright images lose detail.',
  exposure: 'Whether the image appears under- or over-exposed.',
  noise: 'Estimated random variation in pixel values. High noise can mimic or obscure texture.',
  clipping: 'Percentage of saturated pixels at dynamic extremes (0 or 255). Lower clipping preserves more tissue detail.',
  snr: 'Signal-to-noise ratio. Higher values indicate a cleaner image.',
  snrdb: 'Signal-to-noise ratio in decibels. Higher values indicate lower relative noise.',
  resolution: 'Pixel dimensions of the image. Low resolution limits the detail available to the model.',
  entropy: 'A measure of information content in the pixel intensities.',
  rotation: 'Estimated tilt of the patient or image. Strong rotation can distort anatomy.',
  positioning: 'How well the chest appears centred and aligned in the frame.',
  lungcoverage: 'How much of the lung fields appears to be visible in the frame.',
  artifacts: 'Whether foreign objects or processing artefacts were detected.',
  overallscore: 'A combined quality score produced by the backend.',
  qualityscore: 'A combined quality score produced by the backend.',
  compositescore: 'Weighted composite quality score based on six physical and signal indicators.',
};

const UNCERTAINTY: Record<string, string> = {
  predictiveentropy:
    'Total uncertainty in the prediction. Higher values mean the model spreads probability more evenly.',
  entropy: 'Total uncertainty in the prediction. Higher values mean less decisive output.',
  mutualinformation:
    'Epistemic uncertainty: disagreement between repeated stochastic passes. Higher values suggest the model is unsure because of limited knowledge.',
  epistemic: 'Uncertainty about the model itself. It can be reduced with more or better training data.',
  aleatoric: 'Uncertainty inherent in the data, such as noise or ambiguous appearance.',
  variance: 'Spread of predictions across repeated stochastic passes.',
  std: 'Standard deviation of predictions across repeated stochastic passes.',
  standarddeviation: 'Standard deviation of predictions across repeated stochastic passes.',
  meanuncertainty: 'Average uncertainty across the 14 findings.',
  maxuncertainty: 'Largest uncertainty value across the 14 findings.',
  confidence: 'Model-reported confidence. This is not the same as being correct.',
};

const CALIBRATION: Record<string, string> = {
  ece: 'Expected Calibration Error: the average gap between predicted confidence and observed frequency. Lower is better; 0 is perfect.',
  brier:
    'Brier score: mean squared difference between predicted probabilities and outcomes. Lower is better.',
  nll: 'Negative log-likelihood: penalises confident wrong predictions heavily. Lower is better.',
};

export const CALIBRATION_EXPLANATIONS = CALIBRATION;

export const EXPLANATION_METRICS = {
  stability:
    'Explanation stability describes how consistent the highlighted regions remain when the input is slightly perturbed. Higher stability means the explanation changes less under small changes.',
  faithfulness:
    'Explanation faithfulness describes how much the model output changes when the highlighted regions are altered or removed. It tests whether the explanation reflects what the model actually used.',
} as const;

export function qualityMetricExplanation(key: string): string | undefined {
  return QUALITY[normalizeKey(key)];
}

export function uncertaintyMetricExplanation(key: string): string | undefined {
  return UNCERTAINTY[normalizeKey(key)];
}

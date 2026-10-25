/**
 * Illustrative SAMPLE values for the landing-page product preview.
 * They are not model output and are labelled as sample data wherever shown.
 */
export interface PreviewRow {
  finding: string;
  calibrated: number;
  raw: number;
}

export const PREVIEW_ROWS: PreviewRow[] = [
  { finding: 'Effusion', calibrated: 0.58, raw: 0.71 },
  { finding: 'Atelectasis', calibrated: 0.34, raw: 0.42 },
  { finding: 'Cardiomegaly', calibrated: 0.12, raw: 0.15 },
  { finding: 'Infiltration', calibrated: 0.09, raw: 0.11 },
  { finding: 'Pneumonia', calibrated: 0.05, raw: 0.06 },
];

export const PREVIEW_FACTS: Array<{ label: string; value: string }> = [
  { label: 'Image quality', value: 'Adequate' },
  { label: 'Out-of-distribution', value: 'In distribution' },
  { label: 'Mean uncertainty', value: '0.071' },
  { label: 'ECE', value: '0.031' },
];

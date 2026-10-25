import { formatNumber } from '@/lib/format';
import type { ReportMetric } from '@/types/analysis';

export function formatMetricValue(metric: ReportMetric): string {
  const { value, unit, key } = metric;
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string') return unit ? `${value} ${unit}` : value;
  if (Number.isInteger(value)) return unit ? `${value} ${unit}` : String(value);

  let digits = 3;
  if (key.includes('variance') || (Math.abs(value) < 0.01 && value !== 0)) {
    digits = 6;
  } else if (key.includes('std') || key.includes('rms')) {
    digits = 5;
  }

  const text = formatNumber(value, digits);
  return unit ? `${text} ${unit}` : text;
}

import { FlaskConical } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';

/** Marks any element that shows sample (non-model) data. */
export function DemoBadge({ label = 'Sample data' }: { label?: string }) {
  return (
    <Badge tone="outline" icon={<FlaskConical aria-hidden="true" className="h-3 w-3" />}>
      {label}
    </Badge>
  );
}

import { Check } from 'lucide-react';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { Spinner } from '@/components/ui/Spinner';
import { cn } from '@/lib/cn';
import { STAGES } from '@/hooks/useAnalysisRun';

interface ProcessingStagesProps {
  stageIndex: number;
  uploadPercent: number | null;
  onCancel: () => void;
}

/** Progress display. The upload percentage is real; the stage list is visual only. */
export function ProcessingStages({ stageIndex, uploadPercent, onCancel }: ProcessingStagesProps) {
  const uploading = uploadPercent !== null && uploadPercent < 100;

  return (
    <div role="status" aria-live="polite" className="rounded-lg border border-line bg-surface p-5 sm:p-6">
      <div className="mb-4 flex items-center justify-between gap-4">
        <p className="text-sm font-medium">{uploading ? 'Uploading image' : 'Analysing image'}</p>
        {uploadPercent !== null && uploading && <p className="tabular text-xs text-muted">{uploadPercent}%</p>}
      </div>
      <ProgressBar
        label={uploading ? 'Upload progress' : 'Analysis in progress'}
        value={uploading ? (uploadPercent ?? 0) : undefined}
      />

      <ol className="mt-6 space-y-3">
        {STAGES.map((stage, index) => {
          const done = index < stageIndex;
          const active = index === stageIndex;
          return (
            <li
              key={stage.id}
              className={cn('flex items-center gap-3 text-sm', done || active ? 'text-ink' : 'text-muted')}
              aria-current={active ? 'step' : undefined}
            >
              <span
                className={cn(
                  'inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[0.625rem]',
                  done ? 'border-ink bg-ink text-surface' : active ? 'border-ink' : 'border-line',
                )}
              >
                {done ? (
                  <Check aria-hidden="true" className="h-3 w-3" />
                ) : active ? (
                  <Spinner className="h-3 w-3" />
                ) : (
                  index + 1
                )}
              </span>
              {stage.label}
              {done && <span className="sr-only"> (shown as complete)</span>}
            </li>
          );
        })}
      </ol>

      <p className="mt-5 text-xs leading-relaxed text-muted">
        Stage markers indicate that work is in progress; they are not reported by the analysis service.
      </p>

      <button
        type="button"
        onClick={onCancel}
        className="mt-4 rounded text-sm font-medium text-ink underline underline-offset-4 hover:text-charcoal"
      >
        Cancel analysis
      </button>
    </div>
  );
}

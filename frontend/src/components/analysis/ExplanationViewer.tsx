import { useState } from 'react';
import { ImageOff, Maximize2 } from 'lucide-react';
import { Dialog } from '@/components/ui/Dialog';
import { Skeleton } from '@/components/ui/Skeleton';
import { EXPLANATION_METRICS } from '@/lib/explanations';
import { formatNumber } from '@/lib/format';
import { useArtifactUrl } from '@/hooks/useArtifactUrl';
import { ORIGINAL_ARTIFACT_NAME } from '@/services/endpoints';
import type { ExplanationArtifact, ExplanationAssessment, ExplanationMetric } from '@/types/analysis';
import { Unavailable } from './SectionShell';

interface ArtifactImageProps {
  analysisId: string;
  artifactName: string | undefined;
  alt: string;
  className?: string;
}

/** Loads an artifact through the API client and renders it, with loading and failure states. */
export function ArtifactImage({ analysisId, artifactName, alt, className }: ArtifactImageProps) {
  const { url, loading, error } = useArtifactUrl(analysisId, artifactName);

  if (loading) {
    return (
      <div role="status" className="relative aspect-square w-full">
        <span className="sr-only">Loading image…</span>
        <Skeleton className="absolute inset-0 rounded-none" />
      </div>
    );
  }
  if (error || !url) {
    return (
      <div className="flex aspect-square w-full flex-col items-center justify-center gap-2 bg-subtle px-4 text-center text-xs text-muted">
        <ImageOff aria-hidden="true" className="h-5 w-5" />
        Image unavailable
      </div>
    );
  }
  return <img src={url} alt={alt} className={className ?? 'aspect-square w-full object-contain'} />;
}

function MetricCard({ title, metric, explanation }: { title: string; metric: ExplanationMetric | undefined; explanation: string }) {
  return (
    <div className="border-t border-line pt-4">
      <h4 className="text-sm font-medium">{title}</h4>
      {metric ? (
        <>
          <p className="tabular mt-1 text-xl font-semibold tracking-tight">
            {typeof metric.value === 'number' ? formatNumber(metric.value, 3) : metric.value}
          </p>
          {metric.metric && <p className="mt-0.5 text-xs text-muted">Measured with: {metric.metric}</p>}
          <p className="mt-2 text-[0.8125rem] leading-relaxed text-muted">{metric.description ?? explanation}</p>
        </>
      ) : (
        <p className="mt-1 text-sm text-muted">Not calculated for this analysis.</p>
      )}
    </div>
  );
}

interface ExplanationViewerProps {
  analysisId: string;
  explanation: ExplanationAssessment | undefined;
  /** Whether the displayed images are sample illustrations. */
  isDemo: boolean;
}

export function ExplanationViewer({ analysisId, explanation, isDemo }: ExplanationViewerProps) {
  const [active, setActive] = useState<ExplanationArtifact | null>(null);

  if (!explanation) {
    return <Unavailable>Explainability results were not returned for this analysis. They were not evaluated.</Unavailable>;
  }

  return (
    <div className="space-y-8">
      {explanation.method && (
        <p className="text-sm">
          <span className="text-muted">Method: </span>
          <span className="font-medium">{explanation.method}</span>
        </p>
      )}

      {explanation.artifacts.length > 0 ? (
        <ul className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {explanation.artifacts.map((artifact) => (
            <li key={artifact.name}>
              <figure>
                <div className="group relative overflow-hidden rounded-lg border border-line bg-dark">
                  <ArtifactImage
                    analysisId={analysisId}
                    artifactName={artifact.name}
                    alt={`${artifact.label}. Model explanation overlay, not an original image.`}
                  />
                  <span className="absolute left-2 top-2 rounded bg-ink/80 px-1.5 py-0.5 text-[0.6875rem] font-medium text-surface">
                    Generated explanation{isDemo ? ' · sample' : ''}
                  </span>
                  <button
                    type="button"
                    onClick={() => setActive(artifact)}
                    aria-label={`Enlarge ${artifact.label}`}
                    className="absolute bottom-2 right-2 inline-flex h-8 w-8 items-center justify-center rounded-md bg-surface/90 text-ink opacity-0 transition-opacity hover:bg-surface focus-visible:opacity-100 group-hover:opacity-100 max-sm:opacity-100"
                  >
                    <Maximize2 aria-hidden="true" className="h-3.5 w-3.5" />
                  </button>
                </div>
                <figcaption className="mt-2.5">
                  <p className="text-sm font-medium">{artifact.label}</p>
                  {artifact.caption && <p className="mt-0.5 text-xs leading-relaxed text-muted">{artifact.caption}</p>}
                </figcaption>
              </figure>
            </li>
          ))}
        </ul>
      ) : (
        <Unavailable>No explanation images were returned for this analysis.</Unavailable>
      )}

      <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2">
        <MetricCard title="Explanation stability" metric={explanation.stability} explanation={EXPLANATION_METRICS.stability} />
        <MetricCard title="Explanation faithfulness" metric={explanation.faithfulness} explanation={EXPLANATION_METRICS.faithfulness} />
      </div>

      <div className="rounded-md border border-line bg-subtle/50 px-4 py-3.5 text-[0.8125rem] leading-relaxed text-muted space-y-2">
        <p>
          <strong className="font-medium text-charcoal">Spatial Resolution & Architecture:</strong>{' '}
          DenseNet-121&rsquo;s final convolutional layer (<code className="font-mono text-xs">features.norm5</code>) operates at a coarse{' '}
          <strong className="font-medium text-charcoal">7 × 7 spatial grid</strong> (~32 × 32 pixels per receptive element on 224 × 224 inputs). Standalone heatmaps represent bilinear interpolation and Gaussian smoothing of this grid, producing regional attribution rather than sharp anatomical boundary segmentations.
        </p>
        <p>
          Highlighted regions show where the model&rsquo;s logit was most sensitive. They are{' '}
          <strong className="font-medium text-charcoal">model explanations, not proof of pathology</strong>. Attribution validity is corroborated by the quantitative stability (Spearman rank correlation under perturbation) and faithfulness (prediction drop under regional occlusion) metrics above.
        </p>
      </div>

      <Dialog
        open={active !== null}
        onClose={() => setActive(null)}
        title={active?.label ?? 'Explanation'}
        description="Generated explanation overlay. Compare with the original image in the report."
        size="xl"
      >
        {active && (
          <div className="grid gap-px bg-line md:grid-cols-2">
            <figure className="bg-surface p-4">
              <div className="overflow-hidden rounded-md bg-dark">
                <ArtifactImage
                  analysisId={analysisId}
                  artifactName={ORIGINAL_ARTIFACT_NAME}
                  alt="Original uploaded X-ray"
                  className="max-h-[70vh] w-full object-contain"
                />
              </div>
              <figcaption className="mt-2 text-xs text-muted">Original image</figcaption>
            </figure>
            <figure className="bg-surface p-4">
              <div className="overflow-hidden rounded-md bg-dark">
                <ArtifactImage
                  analysisId={analysisId}
                  artifactName={active.name}
                  alt={`${active.label}. Generated explanation overlay.`}
                  className="max-h-[70vh] w-full object-contain"
                />
              </div>
              <figcaption className="mt-2 text-xs text-muted">
                Generated explanation{isDemo ? ' (sample illustration)' : ''}
              </figcaption>
            </figure>
          </div>
        )}
      </Dialog>
    </div>
  );
}

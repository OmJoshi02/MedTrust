import { useMemo } from 'react';
import {
  Activity,
  ArrowRight,
  FlaskConical,
  Gauge,
  Layers,
  ListChecks,
  ScanEye,
  ShieldAlert,
  SlidersHorizontal,
  Upload,
  type LucideIcon,
} from 'lucide-react';
import { PageContainer } from '@/components/layout/PageContainer';
import { ButtonAnchor, ButtonLink } from '@/components/ui/Button';
import { sampleRadiographSvg } from '@/demo/sampleImages';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';
import { formatPercent } from '@/lib/format';
import { PREVIEW_FACTS, PREVIEW_ROWS } from './home/previewData';

const WORKFLOW: Array<{ icon: LucideIcon; title: string; body: string }> = [
  {
    icon: Upload,
    title: 'Upload an X-ray',
    body: 'Select a frontal chest radiograph in PNG, JPG or WebP format. Nothing is analysed until you confirm.',
  },
  {
    icon: ListChecks,
    title: 'Review disease predictions',
    body: 'See probabilities for 14 findings, with raw and calibrated values shown side by side.',
  },
  {
    icon: Gauge,
    title: 'Examine uncertainty and reliability',
    body: 'Check how settled the model is, and whether the image resembles the data it was developed on.',
  },
  {
    icon: ScanEye,
    title: 'Inspect explainability and image quality',
    body: 'View Grad-CAM style overlays and quality metrics to judge how much weight a result deserves.',
  },
];

const CAPABILITIES: Array<{ icon: LucideIcon; title: string; body: string }> = [
  {
    icon: Layers,
    title: 'Multi-label detection',
    body: 'A single radiograph can show several findings at once, so each of the 14 categories receives its own probability rather than one winning label.',
  },
  {
    icon: Gauge,
    title: 'Image-quality assessment',
    body: 'Sharpness, contrast, exposure and related measures indicate whether an image is suitable before its predictions are trusted.',
  },
  {
    icon: Activity,
    title: 'Uncertainty analysis',
    body: 'Uncertainty estimates describe how settled a model output is, separating a confident answer from a guess.',
  },
  {
    icon: ShieldAlert,
    title: 'Out-of-distribution detection',
    body: 'Flags images that look unlike the development data, where predictions are least reliable.',
  },
  {
    icon: SlidersHorizontal,
    title: 'Calibration',
    body: 'Calibrated probabilities are compared with raw model outputs, with ECE, Brier score and reliability diagrams when available.',
  },
  {
    icon: ScanEye,
    title: 'Explainability',
    body: 'Heatmap overlays show which regions influenced an output, alongside stability and faithfulness checks when calculated.',
  },
];

export function HomePage() {
  useDocumentTitle(null);
  const sampleImage = useMemo(
    () => `data:image/svg+xml;utf8,${encodeURIComponent(sampleRadiographSvg(2026))}`,
    [],
  );

  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div aria-hidden="true" className="bg-grid absolute inset-0 -z-10" />
        <PageContainer className="pb-20 pt-16 sm:pb-28 sm:pt-24">
          <p className="eyebrow mb-6">Trust-aware explainable AI &middot; Chest X-ray research</p>
          <h1 className="display max-w-4xl">Understand the prediction. Evaluate its reliability.</h1>
          <p className="lead mt-7 max-w-2xl">
            MedTrust is a research application for multi-label chest X-ray disease detection. Alongside each
            prediction it reports uncertainty, image quality, calibration and explainability, so you can judge how far
            a result should be trusted.
          </p>
          <div className="mt-10 flex flex-wrap gap-3">
            <ButtonLink to="/analyze" size="lg" trailingIcon={<ArrowRight aria-hidden="true" className="h-4 w-4" />}>
              Analyze an X-Ray
            </ButtonLink>
            <ButtonAnchor href="#preview" variant="secondary" size="lg">
              Explore MedTrust
            </ButtonAnchor>
          </div>
        </PageContainer>
      </section>

      {/* Product preview */}
      <section id="preview" aria-labelledby="preview-title" className="scroll-mt-20 pb-24 sm:pb-32">
        <PageContainer>
          <div className="mb-10 flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="eyebrow mb-3">The analysis view</p>
              <h2 id="preview-title" className="h-section max-w-xl">
                A result you can interrogate, not just read.
              </h2>
            </div>
            <p className="inline-flex items-center gap-2 rounded border border-ink/20 px-2.5 py-1 text-xs font-medium text-charcoal">
              <FlaskConical aria-hidden="true" className="h-3.5 w-3.5" />
              Illustration using sample data
            </p>
          </div>

          <div className="glass overflow-hidden rounded-xl shadow-lift">
            <div className="grid lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
              <figure className="relative bg-dark p-4 sm:p-6">
                <img
                  src={sampleImage}
                  alt="Abstract sample illustration of a chest radiograph. It is not a real X-ray."
                  className="mx-auto aspect-square w-full max-w-md rounded object-cover"
                />
                <figcaption className="mt-3 text-center text-xs text-surface/60">
                  Sample illustration, not a real radiograph
                </figcaption>
              </figure>

              <div className="p-6 sm:p-9">
                <div className="mb-5 flex items-baseline justify-between gap-4">
                  <h3 className="text-sm font-semibold">Disease probabilities</h3>
                  <p className="text-xs text-muted">Sample values</p>
                </div>

                <ul className="space-y-4">
                  {PREVIEW_ROWS.map((row) => (
                    <li key={row.finding}>
                      <div className="mb-1.5 flex items-baseline justify-between gap-4 text-sm">
                        <span className="font-medium">{row.finding}</span>
                        <span className="tabular text-muted">
                          <span className="font-medium text-ink">{formatPercent(row.calibrated, 0)}</span> calibrated
                          &middot; {formatPercent(row.raw, 0)} raw
                        </span>
                      </div>
                      <div aria-hidden="true" className="space-y-1">
                        <div className="h-1.5 rounded-full bg-subtle">
                          <div className="h-full rounded-full bg-ink" style={{ width: `${row.calibrated * 100}%` }} />
                        </div>
                        <div className="h-1 rounded-full bg-subtle">
                          <div className="h-full rounded-full bg-ink/30" style={{ width: `${row.raw * 100}%` }} />
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>

                <dl className="mt-8 grid grid-cols-2 gap-x-6 border-t border-line sm:grid-cols-4">
                  {PREVIEW_FACTS.map((fact) => (
                    <div key={fact.label} className="py-4">
                      <dt className="text-xs text-muted">{fact.label}</dt>
                      <dd className="mt-1 text-sm font-medium">{fact.value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          </div>
        </PageContainer>
      </section>

      {/* Workflow */}
      <section aria-labelledby="workflow-title" className="border-y border-line bg-surface/50 py-20 sm:py-28">
        <PageContainer>
          <p className="eyebrow mb-3">Workflow</p>
          <h2 id="workflow-title" className="h-section max-w-2xl">
            Four steps from image to an informed reading.
          </h2>
          <ol className="mt-14 grid gap-x-10 gap-y-12 sm:grid-cols-2 lg:grid-cols-4">
            {WORKFLOW.map((step, index) => (
              <li key={step.title} className="border-t border-ink pt-5">
                <div className="mb-6 flex items-center justify-between">
                  <span className="tabular text-xs font-medium text-muted">{String(index + 1).padStart(2, '0')}</span>
                  <step.icon aria-hidden="true" className="h-5 w-5 text-ink" strokeWidth={1.5} />
                </div>
                <h3 className="text-base font-semibold tracking-tight">{step.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{step.body}</p>
              </li>
            ))}
          </ol>
        </PageContainer>
      </section>

      {/* Capabilities */}
      <section aria-labelledby="capabilities-title" className="on-dark bg-dark py-20 text-surface sm:py-28">
        <PageContainer>
          <div className="grid gap-12 lg:grid-cols-[1fr_2fr] lg:gap-20">
            <div>
              <p className="eyebrow !text-surface/50 mb-3">Research capabilities</p>
              <h2 id="capabilities-title" className="h-section !text-surface">
                Reliability is part of the result.
              </h2>
              <p className="mt-5 max-w-sm text-sm leading-relaxed text-surface/65">
                A probability on its own hides how much it can be trusted. MedTrust treats the evidence around a
                prediction as a first-class output.
              </p>
            </div>
            <ul className="grid gap-x-12 gap-y-10 sm:grid-cols-2">
              {CAPABILITIES.map((item) => (
                <li key={item.title} className="border-t border-white/15 pt-5">
                  <item.icon aria-hidden="true" className="mb-4 h-5 w-5 text-surface" strokeWidth={1.5} />
                  <h3 className="text-base font-semibold tracking-tight !text-surface">{item.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-surface/65">{item.body}</p>
                </li>
              ))}
            </ul>
          </div>
        </PageContainer>
      </section>

      {/* Disclaimer */}
      <section aria-labelledby="disclaimer-title" className="py-20 sm:py-24">
        <PageContainer>
          <div className="grid gap-6 border-y border-ink py-10 lg:grid-cols-[1fr_2fr] lg:gap-20">
            <h2 id="disclaimer-title" className="text-xl font-semibold tracking-tight">
              Medical research disclaimer
            </h2>
            <div className="max-w-prose space-y-3 text-sm leading-relaxed text-charcoal">
              <p>
                MedTrust is an academic research prototype. It is not a medical device, has not been clinically
                validated, and is not intended for diagnosis, screening, treatment or any clinical decision.
              </p>
              <p>
                A predicted probability is a statistical estimate from a model, not a confirmed diagnosis. Highlighted
                regions are model explanations, not proof of pathology. Always rely on a qualified clinician&rsquo;s
                interpretation of imaging.
              </p>
            </div>
          </div>
          <div className="mt-10 flex flex-wrap items-center justify-between gap-4">
            <p className="text-sm text-muted">Ready to see the full report layout?</p>
            <ButtonLink to="/analyze" variant="secondary" trailingIcon={<ArrowRight aria-hidden="true" className="h-4 w-4" />}>
              Analyze an X-Ray
            </ButtonLink>
          </div>
        </PageContainer>
      </section>
    </>
  );
}

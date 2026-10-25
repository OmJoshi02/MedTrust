import { useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, Download, Printer } from 'lucide-react';
import { CalibrationPanel } from '@/components/analysis/CalibrationPanel';
import { DiseaseProbabilityList } from '@/components/analysis/DiseaseProbabilityList';
import { ArtifactImage, ExplanationViewer } from '@/components/analysis/ExplanationViewer';
import { OODPanel } from '@/components/analysis/OODPanel';
import { QualityAssessment } from '@/components/analysis/QualityAssessment';
import { SectionShell, Unavailable } from '@/components/analysis/SectionShell';
import { SimilarityPanel } from '@/components/analysis/SimilarityPanel';
import { UncertaintyPanel } from '@/components/analysis/UncertaintyPanel';
import { DemoBadge } from '@/components/feedback/DemoBadge';
import { Alert } from '@/components/ui/Alert';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Skeleton } from '@/components/ui/Skeleton';
import { PageContainer } from '@/components/layout/PageContainer';
import { ErrorState } from '@/components/feedback/ErrorState';
import { useAsyncData } from '@/hooks/useAsyncData';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';
import { formatBytes, formatDateTime } from '@/lib/format';
import { analysisService, isApiError } from '@/services';
import { ORIGINAL_ARTIFACT_NAME } from '@/services/endpoints';
import { hasPredictions, type AnalysisReport, type AnalysisStatus } from '@/types/analysis';
import { ReportNotFound, UnauthorizedPage } from './ErrorPages';

const STATUS: Record<AnalysisStatus, { label: string; tone: BadgeTone }> = {
  completed: { label: 'Completed', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
  processing: { label: 'Processing', tone: 'warning' },
  queued: { label: 'Queued', tone: 'neutral' },
  unknown: { label: 'Status unknown', tone: 'neutral' },
};

const SECTIONS = [
  { id: 'image', label: 'Uploaded X-ray' },
  { id: 'predictions', label: 'Disease predictions' },
  { id: 'quality', label: 'Image quality' },
  { id: 'uncertainty', label: 'Uncertainty' },
  { id: 'ood', label: 'Out-of-distribution' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'explainability', label: 'Explainability' },
  { id: 'similarity', label: 'Similar cases' },
  { id: 'limitations', label: 'Limitations' },
] as const;

function downloadJson(report: AnalysisReport) {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `medtrust-report-${report.id}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ReportSkeleton() {
  return (
    <PageContainer className="py-14" aria-busy="true">
      <p role="status" className="sr-only">
        Loading report…
      </p>
      <Skeleton className="h-4 w-32" />
      <Skeleton className="mt-4 h-12 w-2/3 max-w-xl" />
      <Skeleton className="mt-6 h-4 w-1/2 max-w-md" />
      <div className="mt-14 grid gap-10 lg:grid-cols-2">
        <Skeleton className="aspect-square w-full" />
        <div className="space-y-4">
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} className="h-6 w-full" />
          ))}
        </div>
      </div>
    </PageContainer>
  );
}

function ReportBody({ report }: { report: AnalysisReport }) {
  const { image } = report;
  const status = STATUS[report.status];
  const failed = report.status === 'failed';
  const incomplete = report.status === 'queued' || report.status === 'processing';

  const metadata = useMemo(() => {
    const rows: Array<{ label: string; value: string }> = [];
    if (image.width !== undefined && image.height !== undefined) {
      rows.push({ label: 'Dimensions', value: `${image.width} × ${image.height} px` });
    }
    if (image.sizeBytes !== undefined && image.sizeBytes > 0) {
      rows.push({ label: 'File size', value: formatBytes(image.sizeBytes) });
    } else {
      rows.push({ label: 'File size', value: 'Not available' });
    }
    return [...rows, ...image.extra];
  }, [image]);

  const positiveFindings = useMemo(() => {
    return (report.predictions || [])
      .filter((p) => {
        const calThr = p.calibratedThreshold ?? p.threshold;
        const rawThr = p.rawThreshold ?? p.threshold;
        const isCalPos =
          p.calibratedFlagged !== undefined && p.calibratedFlagged !== null
            ? p.calibratedFlagged
            : p.calibratedProbability !== null && calThr !== null
            ? p.calibratedProbability >= calThr
            : p.flagged;
        const isRawPos =
          p.rawFlagged !== undefined && p.rawFlagged !== null
            ? p.rawFlagged
            : p.rawProbability !== null && rawThr !== null
            ? p.rawProbability >= rawThr
            : p.flagged;
        return Boolean(isCalPos ?? isRawPos ?? p.flagged);
      })
      .map((p) => p.disease);
  }, [report.predictions]);

  return (
    <PageContainer className="py-10 sm:py-14">
      <Link
        to="/history"
        className="no-print mb-8 inline-flex items-center gap-1.5 rounded text-sm text-muted transition-colors hover:text-ink"
      >
        <ArrowLeft aria-hidden="true" className="h-4 w-4" />
        Recent analyses
      </Link>

      {/* Header */}
      <header className="mb-12">
        <div className="flex flex-wrap items-center gap-3">
          <p className="eyebrow">Analysis report</p>
          <Badge tone={status.tone}>{status.label}</Badge>
          {report.isDemo && <DemoBadge label="Sample data — not model output" />}
        </div>
        <h1 className="h-page mt-4 break-words">{report.filename ?? 'Untitled image'}</h1>

        <dl className="mt-8 grid gap-x-10 gap-y-5 border-y border-line py-5 sm:grid-cols-2 lg:grid-cols-4">
          <div className="min-w-0">
            <dt className="text-xs text-muted">Analysis ID</dt>
            <dd className="tabular mt-1 break-all font-mono text-sm">{report.id}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Date and time</dt>
            <dd className="mt-1 text-sm">{formatDateTime(report.createdAt)}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-xs text-muted">Image file</dt>
            <dd className="mt-1 truncate text-sm" title={report.filename}>
              {report.filename ?? 'Not provided'}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Model</dt>
            <dd className="mt-1 text-sm">
              {report.model?.name ? `${report.model.name}${report.model.version ? ` ${report.model.version}` : ''}` : 'Not provided'}
            </dd>
          </div>
        </dl>

        <div className="no-print mt-5 flex flex-wrap gap-2">
          <Button variant="secondary" size="sm" onClick={() => window.print()} leadingIcon={<Printer aria-hidden="true" className="h-3.5 w-3.5" />}>
            Print or save as PDF
          </Button>
          <Button variant="secondary" size="sm" onClick={() => downloadJson(report)} leadingIcon={<Download aria-hidden="true" className="h-3.5 w-3.5" />}>
            Download data (JSON)
          </Button>
        </div>
      </header>

      {failed && (
        <Alert tone="danger" title="This analysis did not complete." className="mb-10">
          {report.error ?? 'The backend reported a failure without further detail.'}
        </Alert>
      )}
      {incomplete && (
        <Alert tone="warning" title="This analysis is still in progress." className="mb-10">
          Results may be incomplete. Reload this page later to see the finished report.
        </Alert>
      )}
      {report.warnings.length > 0 && (
        <Alert tone="warning" title="Notes about this report" className="mb-10">
          <ul className="list-disc pl-4">
            {report.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </Alert>
      )}
      {report.isDemo && (
        <Alert tone="info" title="You are viewing sample data" className="mb-10">
          Every value and image in this report was generated to preview the layout. None of it was produced by the
          MedTrust model.
        </Alert>
      )}

      <div className="grid gap-12 lg:grid-cols-[13rem_minmax(0,1fr)] lg:gap-16">
        <nav aria-label="Report sections" className="no-print -mx-[var(--gutter)] overflow-x-auto px-[var(--gutter)] lg:mx-0 lg:overflow-visible lg:px-0">
          <ul className="flex gap-1 whitespace-nowrap lg:sticky lg:top-28 lg:flex-col lg:gap-0 lg:border-l lg:border-line">
            {SECTIONS.map((section, index) => (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  className="flex gap-3 rounded px-3 py-1.5 text-sm text-muted transition-colors hover:text-ink lg:-ml-px lg:rounded-none lg:border-l lg:border-transparent lg:py-1.5 lg:pl-4 lg:hover:border-ink"
                >
                  <span className="tabular hidden text-xs lg:inline">{String(index + 1).padStart(2, '0')}</span>
                  {section.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="min-w-0 space-y-14">
          <SectionShell
            id="image"
            number="01"
            title="Uploaded X-ray"
            description="The original image as submitted. Generated explanation images appear separately in the Explainability section."
          >
            <div className="grid gap-8 md:grid-cols-[minmax(0,22rem)_1fr]">
              <figure>
                <div className="overflow-hidden rounded-lg border border-line bg-dark">
                  <ArtifactImage
                    analysisId={report.id}
                    artifactName={image.artifactName ?? ORIGINAL_ARTIFACT_NAME}
                    alt={`Original uploaded X-ray${report.filename ? `: ${report.filename}` : ''}`}
                  />
                </div>
                <figcaption className="mt-2 text-xs text-muted">
                  Original image{report.isDemo ? ' (sample illustration unless you uploaded a file)' : ''}
                </figcaption>
              </figure>
              {metadata.length > 0 ? (
                <dl className="divide-y divide-line self-start border-y border-line">
                  {metadata.map((row) => (
                    <div key={`${row.label}-${row.value}`} className="grid grid-cols-[8rem_1fr] gap-3 py-3 text-sm">
                      <dt className="text-muted">{row.label}</dt>
                      <dd className="min-w-0 break-words font-medium">{row.value}</dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <Unavailable>No image metadata was returned.</Unavailable>
              )}
            </div>
          </SectionShell>

          <SectionShell
            id="predictions"
            number="02"
            title="Disease prediction results"
            description="Probabilities for the 14 findings in standard order. Raw values are the model's direct output; calibrated values have been adjusted to better match observed frequencies."
          >
            {hasPredictions(report) ? (
              <DiseaseProbabilityList predictions={report.predictions} />
            ) : (
              <Unavailable>No disease probabilities were returned for this analysis.</Unavailable>
            )}
            <p className="mt-6 max-w-prose rounded-md border border-line bg-subtle/50 px-4 py-3 text-[0.8125rem] leading-relaxed text-muted">
              A predicted probability is the model&rsquo;s estimate. It is{' '}
              <strong className="font-medium text-charcoal">not a confirmed diagnosis</strong>.
            </p>
          </SectionShell>

          <SectionShell
            id="quality"
            number="03"
            title="Image-quality assessment"
            description="Only metrics returned by the analysis service are shown."
          >
            <QualityAssessment quality={report.quality} />
          </SectionShell>

          <SectionShell
            id="uncertainty"
            number="04"
            title="Uncertainty assessment"
            description="How settled the model output is for this image."
          >
            <UncertaintyPanel uncertainty={report.uncertainty} />
          </SectionShell>

          <SectionShell
            id="ood"
            number="05"
            title="Out-of-distribution assessment"
            description="Whether this image resembles the data the model was developed on."
          >
            <OODPanel ood={report.ood} />
          </SectionShell>

          <SectionShell
            id="calibration"
            number="06"
            title="Calibration"
            description="How well predicted probabilities correspond to observed frequencies."
          >
            <CalibrationPanel calibration={report.calibration} />
          </SectionShell>

          <SectionShell
            id="explainability"
            number="07"
            title="Explainability"
            description="Generated overlays showing where the model output was most sensitive."
          >
            <ExplanationViewer analysisId={report.id} explanation={report.explanation} isDemo={report.isDemo} />
          </SectionShell>

          <SectionShell
            id="similarity"
            number="08"
            title="Case-based similarity"
            description="Top similar reference cases retrieved using DenseNet-121 1024-d visual feature representations and Cosine Similarity."
          >
            <SimilarityPanel
              similarity={report.similarity}
              queryImageUrl={report.image ? `/api/v1/analyses/${encodeURIComponent(report.id)}/artifacts/${report.image.artifactName ?? 'original'}` : undefined}
              queryFilename={report.filename}
              predictedFindings={positiveFindings}
              analysisId={report.id}
            />
          </SectionShell>

          <SectionShell id="limitations" number="09" title="Limitations and disclaimer">
            <div className="max-w-prose space-y-3 text-sm leading-relaxed text-charcoal">
              <p>
                MedTrust is a research prototype. It is not a medical device, has not been clinically validated, and must
                not be used for diagnosis, triage, treatment or any clinical decision.
              </p>
              <ul className="list-disc space-y-1.5 pl-5">
                <li>Probabilities are estimates and can be wrong, including when reported uncertainty is low.</li>
                <li>Predictions are less reliable for images of poor quality or unlike the development data.</li>
                <li>Calibration figures describe the model overall, not this individual image.</li>
                <li>Highlighted regions are model explanations, not proof of pathology.</li>
                <li>Any value labelled &ldquo;not evaluated&rdquo; was not returned by the analysis service.</li>
              </ul>
              <p>A qualified clinician&rsquo;s interpretation of the original imaging must always take precedence.</p>
            </div>
          </SectionShell>
        </div>
      </div>
    </PageContainer>
  );
}

export function ReportPage() {
  const { analysisId = '' } = useParams<{ analysisId: string }>();
  const { data, error, loading, reload } = useAsyncData(
    (signal) => analysisService.getReport(analysisId, signal),
    `report:${analysisId}`,
    analysisId !== '',
  );
  useDocumentTitle(data ? `Report ${data.id}` : 'Report');

  if (loading) return <ReportSkeleton />;

  if (error) {
    if (isApiError(error)) {
      // 404 and 403 are treated alike so another user's report IDs cannot be probed.
      if (error.kind === 'not_found' || error.kind === 'forbidden') return <ReportNotFound />;
      if (error.kind === 'unauthorized') return <UnauthorizedPage />;
    }
    return (
      <PageContainer className="py-20">
        <ErrorState error={error} onRetry={reload} title="The report could not be loaded" />
      </PageContainer>
    );
  }

  if (!data) return <ReportNotFound />;
  return <ReportBody report={data} />;
}

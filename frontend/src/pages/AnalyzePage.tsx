import { useId } from 'react';
import { useNavigate } from 'react-router-dom';
import { ScanSearch } from 'lucide-react';
import { ProcessingStages } from '@/components/analysis/ProcessingStages';
import { ResultsPreview } from '@/components/analysis/ResultsPreview';
import { PageContainer } from '@/components/layout/PageContainer';
import { ImageDropzone } from '@/components/upload/ImageDropzone';
import { ImagePreview } from '@/components/upload/ImagePreview';
import { Alert } from '@/components/ui/Alert';
import { Button, ButtonLink } from '@/components/ui/Button';
import { env } from '@/config/env';
import { useAnalysisRun } from '@/hooks/useAnalysisRun';
import { useAuth } from '@/hooks/useAuth';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';

const ERROR_TITLES: Record<string, string> = {
  network: 'Could not reach the analysis service',
  timeout: 'The analysis timed out',
  unauthorized: 'Sign-in required',
  payload_too_large: 'The file is too large',
  unsupported_media: 'Unsupported file format',
  server: 'The analysis service reported an error',
};

export function AnalyzePage() {
  useDocumentTitle('Analyze X-Ray');
  const run = useAnalysisRun();
  const { status: authStatus } = useAuth();
  const navigate = useNavigate();
  const errorId = useId();
  const running = run.phase === 'running';
  const maxBytes = env.maxUploadMb * 1024 * 1024;

  const handleAnalyze = () => {
    if (authStatus === 'unauthenticated' && !env.demoMode) {
      navigate('/login', { state: { from: '/analyze' } });
      return;
    }
    void run.submit();
  };

  return (
    <PageContainer className="py-12 sm:py-16">
      <header className="mb-10 max-w-2xl">
        <p className="eyebrow mb-3">Analysis workspace</p>
        <h1 className="h-page">Analyze a chest X-ray</h1>
        <p className="lead mt-4">
          Select a frontal chest radiograph. MedTrust returns disease probabilities together with image-quality,
          uncertainty, out-of-distribution, calibration and explainability information.
        </p>
      </header>

      {env.demoMode && (
        <Alert tone="info" title="Demo mode is on" className="mb-8 max-w-3xl">
          Your image stays in this browser and is not analysed by any model. The result you will see is generated sample
          data created only to preview the interface.
        </Alert>
      )}

      {authStatus === 'unauthenticated' && !env.demoMode && (
        <Alert
          tone="warning"
          title="Sign-in required for live analysis"
          className="mb-8 max-w-3xl"
          action={
            <ButtonLink to="/login" size="sm" variant="secondary">
              Sign In
            </ButtonLink>
          }
        >
          You are currently in guest mode. Live neural analysis requires investigator authentication. Sign in to execute the clinical diagnostic pipeline.
        </Alert>
      )}

      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-14">
        <section aria-labelledby="upload-heading">
          <h2 id="upload-heading" className="mb-4 text-sm font-semibold">
            1. Select an image
          </h2>

          {run.file ? (
            <ImagePreview
              file={run.file}
              onRemove={run.clearFile}
              onReplace={(next) => void run.selectFile(next)}
              disabled={running}
            />
          ) : (
            <ImageDropzone
              onFileSelected={(file) => void run.selectFile(file)}
              maxBytes={maxBytes}
              describedBy={run.fileError ? errorId : undefined}
              hasError={Boolean(run.fileError)}
            />
          )}

          {run.fileError && (
            <div id={errorId} className="mt-4">
              <Alert tone="danger" title="This file cannot be used">
                {run.fileError}
              </Alert>
            </div>
          )}

          <div className="mt-6 flex flex-wrap items-center gap-4">
            <Button
              size="lg"
              disabled={!run.file}
              loading={running}
              onClick={handleAnalyze}
              leadingIcon={<ScanSearch aria-hidden="true" className="h-4 w-4" />}
            >
              {running ? 'Analysing…' : 'Analyze Image'}
            </Button>
            {!run.file && <p className="text-sm text-muted">Select an image to enable analysis.</p>}
          </div>

          <p className="mt-6 max-w-md text-xs leading-relaxed text-muted">
            Use only images you are permitted to process. Do not upload identifiable patient information.
            {env.demoMode ? '' : ' Images are sent to the configured MedTrust backend for analysis.'}
          </p>
        </section>

        <section aria-labelledby="results-heading" aria-live="polite">
          <h2 id="results-heading" className="mb-4 text-sm font-semibold">
            2. Review the results
          </h2>

          {run.phase === 'idle' && (
            <div className="flex min-h-[20rem] flex-col items-center justify-center rounded-lg border border-line bg-subtle/40 px-8 py-12 text-center">
              <p className="text-sm font-medium">No analysis yet</p>
              <p className="mt-1.5 max-w-xs text-sm leading-relaxed text-muted">
                Results appear here after you select an image and choose Analyze Image. Nothing is analysed until you do.
              </p>
            </div>
          )}

          {running && (
            <ProcessingStages stageIndex={run.stageIndex} uploadPercent={run.uploadPercent} onCancel={run.cancel} />
          )}

          {run.phase === 'error' && run.error && (
            <Alert
              tone="danger"
              title={ERROR_TITLES[run.error.kind] ?? 'The analysis could not be completed'}
              action={
                <Button size="sm" variant="secondary" onClick={() => void run.submit()}>
                  Try again
                </Button>
              }
            >
              {run.error.message}
            </Alert>
          )}

          {run.phase === 'success' && run.report && <ResultsPreview report={run.report} />}
        </section>
      </div>
    </PageContainer>
  );
}

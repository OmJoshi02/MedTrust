import { CloudOff, FileQuestion, LockKeyhole, TriangleAlert } from 'lucide-react';
import { Link, isRouteErrorResponse, useNavigate, useRouteError } from 'react-router-dom';
import { StatusPage } from '@/components/feedback/StatusPage';
import { Button, ButtonLink } from '@/components/ui/Button';
import { describeError } from '@/services';

export function NotFoundPage() {
  return (
    <StatusPage
      documentTitle="Page not found"
      code="404"
      title="This page could not be found."
      description="The address may be mistyped, or the page may have been moved. Nothing has been lost; you can pick up from one of the pages below."
      actions={
        <>
          <ButtonLink to="/">Go to the home page</ButtonLink>
          <ButtonLink to="/analyze" variant="secondary">
            Analyze an X-Ray
          </ButtonLink>
        </>
      }
    />
  );
}

export function UnauthorizedPage() {
  return (
    <StatusPage
      documentTitle="Access restricted"
      icon={<LockKeyhole aria-hidden="true" className="h-5 w-5" />}
      title="You do not have access to this page."
      description="Sign in with the account that owns this content. If you are signed in, this item may belong to another user or may no longer exist."
      actions={
        <>
          <ButtonLink to="/login">Log in</ButtonLink>
          <ButtonLink to="/" variant="secondary">
            Go to the home page
          </ButtonLink>
        </>
      }
    />
  );
}

export function ApiUnavailablePage({ onRetry }: { onRetry?: () => void }) {
  return (
    <StatusPage
      documentTitle="Service unavailable"
      icon={<CloudOff aria-hidden="true" className="h-5 w-5" />}
      title="The MedTrust service is unavailable."
      description={
        <>
          The application could not reach its analysis service. Check your connection and that the backend is running at
          the address set in <code className="font-mono text-sm">VITE_API_BASE_URL</code>, or enable demo mode to review
          the interface without a backend.
        </>
      }
      actions={
        <>
          {onRetry && <Button onClick={onRetry}>Try again</Button>}
          <ButtonLink to="/" variant="secondary">
            Go to the home page
          </ButtonLink>
        </>
      }
    />
  );
}

/** Route-level error boundary for unexpected runtime or loader errors. */
export function AppErrorPage() {
  const error = useRouteError();
  const navigate = useNavigate();

  const detail = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : error instanceof Error
      ? describeError(error)
      : 'An unknown error occurred.';

  if (isRouteErrorResponse(error) && error.status === 404) return <NotFoundPage />;

  return (
    <StatusPage
      documentTitle="Something went wrong"
      icon={<TriangleAlert aria-hidden="true" className="h-5 w-5" />}
      title="Something went wrong."
      description={
        <>
          The application hit an unexpected problem. Reloading usually resolves it.{' '}
          <span className="mt-3 block rounded-md border border-line bg-subtle/60 px-3 py-2 font-mono text-xs text-muted">
            {detail}
          </span>
        </>
      }
      actions={
        <>
          <Button onClick={() => window.location.reload()}>Reload the page</Button>
          <Button variant="secondary" onClick={() => navigate('/', { replace: true })}>
            Go to the home page
          </Button>
        </>
      }
    />
  );
}

export function ReportNotFound() {
  return (
    <StatusPage
      documentTitle="Report not found"
      icon={<FileQuestion aria-hidden="true" className="h-5 w-5" />}
      title="This report could not be found."
      description={
        <>
          It may not exist, or it may belong to another account. Your own analyses are listed under{' '}
          <Link className="font-medium text-ink underline underline-offset-4" to="/history">
            Recent Analyses
          </Link>
          .
        </>
      }
      actions={
        <>
          <ButtonLink to="/analyze">Analyze an X-Ray</ButtonLink>
          <ButtonLink to="/history" variant="secondary">
            Recent Analyses
          </ButtonLink>
        </>
      }
    />
  );
}

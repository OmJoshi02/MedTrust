import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { Skeleton } from '@/components/ui/Skeleton';
import { PageContainer } from './PageContainer';
import { useAuth } from '@/hooks/useAuth';

/**
 * Client-side route guard. This is a convenience for the interface only:
 * the backend must enforce authorization on every request.
 */
export function ProtectedRoute() {
  const { status } = useAuth();
  const location = useLocation();

  if (status === 'loading') {
    return (
      <PageContainer className="py-20" aria-busy="true">
        <p className="sr-only" role="status">
          Restoring your session…
        </p>
        <Skeleton className="h-10 w-64" />
        <Skeleton className="mt-6 h-4 w-full max-w-xl" />
        <Skeleton className="mt-3 h-4 w-full max-w-md" />
      </PageContainer>
    );
  }

  if (status === 'unauthenticated') {
    return <Navigate to="/login" replace state={{ from: `${location.pathname}${location.search}` }} />;
  }

  return <Outlet />;
}

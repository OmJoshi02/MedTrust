import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowRight, ArrowUp, ChevronLeft, ChevronRight, FileSearch, Search, Trash2 } from 'lucide-react';
import { AnalysisStatusBadge } from '@/components/analysis/AnalysisStatusBadge';
import { DemoBadge } from '@/components/feedback/DemoBadge';
import { EmptyState } from '@/components/feedback/EmptyState';
import { ErrorState } from '@/components/feedback/ErrorState';
import { PageContainer } from '@/components/layout/PageContainer';
import { Alert } from '@/components/ui/Alert';
import { Button, ButtonLink } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Skeleton } from '@/components/ui/Skeleton';
import { env } from '@/config/env';
import { useAsyncData } from '@/hooks/useAsyncData';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';
import { formatDateTime, formatPercent } from '@/lib/format';
import { analysisService } from '@/services';
import type { SortField, SortOrder } from '@/types/api';
import { formatDiseaseName, type AnalysisSummary } from '@/types/analysis';

const PAGE_SIZE = 10;

const SORT_LABELS: Record<SortField, string> = {
  created_at: 'Date',
  filename: 'File name',
  status: 'Status',
};

export function HistoryPage() {
  useDocumentTitle('Recent Analyses');

  const [query, setQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortField>('created_at');
  const [order, setOrder] = useState<SortOrder>('desc');
  const [page, setPage] = useState(1);
  const debouncedQuery = useDebouncedValue(query.trim(), 300);

  const { data, error, loading, reload } = useAsyncData(
    (signal) => analysisService.list({ page, pageSize: PAGE_SIZE, query: debouncedQuery, sortBy, order }, signal),
    `history:${page}:${debouncedQuery}:${sortBy}:${order}`,
  );

  const searching = debouncedQuery !== '';
  const totalPages = data?.total ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : null;

  const [itemToDelete, setItemToDelete] = useState<AnalysisSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleConfirmDelete = async () => {
    if (!itemToDelete) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await analysisService.delete(itemToDelete.id);
      setItemToDelete(null);
      await reload();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete analysis.');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <PageContainer className="py-12 sm:py-16">
      <header className="mb-10 flex flex-wrap items-end justify-between gap-6">
        <div className="max-w-2xl">
          <p className="eyebrow mb-3">Your analyses</p>
          <h1 className="h-page">Recent analyses</h1>
          <p className="lead mt-4">Previous analyses saved to your account. Select one to open its full report.</p>
        </div>
        <ButtonLink to="/analyze" trailingIcon={<ArrowRight aria-hidden="true" className="h-4 w-4" />}>
          New analysis
        </ButtonLink>
      </header>

      {env.demoMode && (
        <Alert tone="info" title="These records are sample data" className="mb-8 max-w-3xl">
          Demo mode lists generated sample analyses (and any you create in this browser session). They were not produced
          by the MedTrust model.
        </Alert>
      )}

      <div className="mb-6 flex flex-wrap items-end gap-4">
        <div className="min-w-[14rem] flex-1 sm:max-w-sm">
          <label htmlFor="history-search" className="mb-1.5 block text-sm font-medium">
            Search
          </label>
          <div className="relative">
            <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
            <input
              id="history-search"
              type="search"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(1);
              }}
              placeholder="File name, ID or finding"
              className="h-11 w-full rounded-md border border-line bg-surface pl-9 pr-3 text-sm placeholder:text-muted/70 hover:border-ink/40 focus:border-ink"
            />
          </div>
        </div>

        <div>
          <label htmlFor="history-sort" className="mb-1.5 block text-sm font-medium">
            Sort by
          </label>
          <div className="flex gap-2">
            <select
              id="history-sort"
              value={sortBy}
              onChange={(e) => {
                setSortBy(e.target.value as SortField);
                setPage(1);
              }}
              className="h-11 rounded-md border border-line bg-surface px-3 text-sm hover:border-ink/40 focus:border-ink"
            >
              {(Object.keys(SORT_LABELS) as SortField[]).map((field) => (
                <option key={field} value={field}>
                  {SORT_LABELS[field]}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => {
                setOrder((o) => (o === 'asc' ? 'desc' : 'asc'));
                setPage(1);
              }}
              aria-label={order === 'asc' ? 'Sorted ascending. Switch to descending.' : 'Sorted descending. Switch to ascending.'}
              className="inline-flex h-11 w-11 items-center justify-center rounded-md border border-line bg-surface hover:border-ink/40"
            >
              {order === 'asc' ? <ArrowUp aria-hidden="true" className="h-4 w-4" /> : <ArrowDown aria-hidden="true" className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>

      <div aria-live="polite" className="border-y border-line">
        {loading && (
          <div role="status" aria-busy="true">
            <span className="sr-only">Loading analyses…</span>
            {Array.from({ length: 5 }, (_, i) => (
              <div key={i} className="border-b border-line/70 py-5 last:border-0">
                <Skeleton className="h-5 w-2/5" />
                <Skeleton className="mt-3 h-4 w-3/5" />
              </div>
            ))}
          </div>
        )}

        {!loading && error !== undefined && <ErrorState error={error} onRetry={reload} title="Your analyses could not be loaded" />}

        {!loading && !error && data && data.items.length === 0 && (
          <EmptyState
            icon={<FileSearch aria-hidden="true" className="h-5 w-5" />}
            title={searching ? 'No analyses match your search' : 'No analyses yet'}
            description={
              searching
                ? 'Try a different file name, ID or finding, or clear the search.'
                : 'Analyses you run while signed in will be listed here.'
            }
            action={
              searching ? (
                <Button variant="secondary" onClick={() => setQuery('')}>
                  Clear search
                </Button>
              ) : (
                <ButtonLink to="/analyze">Analyze an X-Ray</ButtonLink>
              )
            }
          />
        )}

        {!loading && !error && data && data.items.length > 0 && (
          <ul>
            {data.items.map((item) => (
              <li key={item.id} className="group/row flex items-center justify-between border-b border-line/70 last:border-0 hover:bg-surface/70">
                <Link
                  to={`/report/${encodeURIComponent(item.id)}`}
                  className="group grid min-w-0 flex-1 items-center gap-x-6 gap-y-2 py-5 md:grid-cols-[minmax(0,2fr)_minmax(0,1.4fr)_auto_1.5rem] md:px-3"
                >
                  <div className="min-w-0">
                    <p className="truncate font-medium text-ink" title={item.filename}>
                      {item.filename ?? 'Untitled image'}
                    </p>
                    <p className="mt-1 text-xs text-muted">
                      {formatDateTime(item.createdAt)} &middot; <span className="font-mono">{item.id}</span>
                    </p>
                  </div>
                  <p className="text-sm text-charcoal">
                    {item.topFinding ? (
                      <>
                        <span className="text-muted">Highest: </span>
                        {formatDiseaseName(item.topFinding.disease)}{' '}
                        <span className="tabular text-muted">
                          {formatPercent(item.topFinding.probability, 0)} {item.topFinding.calibrated ? 'calibrated' : 'raw'}
                        </span>
                      </>
                    ) : (
                      <span className="text-muted">No summary available</span>
                    )}
                  </p>
                  <div className="flex flex-wrap items-center gap-2">
                    <AnalysisStatusBadge status={item.status} />
                    {item.isDemo && <DemoBadge label="Sample" />}
                  </div>
                  <ArrowRight aria-hidden="true" className="hidden h-4 w-4 text-muted transition-transform group-hover:translate-x-0.5 group-hover:text-ink md:block" />
                  <span className="sr-only">Open report</span>
                </Link>

                <div className="flex items-center px-2 sm:px-3">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setDeleteError(null);
                      setItemToDelete(item);
                    }}
                    className="inline-flex h-9 w-9 items-center justify-center rounded-md text-muted transition-colors hover:bg-red-50 hover:text-red-600 focus:outline-none focus:ring-2 focus:ring-red-500/20 dark:hover:bg-red-950/40 dark:hover:text-red-400"
                    aria-label={`Delete analysis ${item.filename ?? item.id}`}
                    title="Delete this analysis"
                  >
                    <Trash2 aria-hidden="true" className="h-4 w-4" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {!loading && !error && data && (data.items.length > 0 || page > 1) && (
        <nav aria-label="Pagination" className="mt-6 flex items-center justify-between gap-4">
          <p className="tabular text-sm text-muted">
            Page {data.page}
            {totalPages ? ` of ${totalPages}` : ''}
            {data.total !== null ? ` · ${data.total} ${data.total === 1 ? 'analysis' : 'analyses'}` : ''}
          </p>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              leadingIcon={<ChevronLeft aria-hidden="true" className="h-4 w-4" />}
            >
              Previous
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={!data.hasMore}
              onClick={() => setPage((p) => p + 1)}
              trailingIcon={<ChevronRight aria-hidden="true" className="h-4 w-4" />}
            >
              Next
            </Button>
          </div>
        </nav>
      )}

      <Dialog
        open={itemToDelete !== null}
        onClose={() => {
          if (!deleting) {
            setItemToDelete(null);
            setDeleteError(null);
          }
        }}
        title="Delete analysis"
        description="Permanently remove this analysis record."
      >
        <div className="flex flex-col gap-6 p-6 sm:p-7">
          <p className="text-sm leading-relaxed text-charcoal">
            Are you sure you want to delete{' '}
            <strong className="font-semibold text-ink">
              {itemToDelete?.filename ?? `Analysis #${itemToDelete?.id}`}
            </strong>
            ? This will permanently remove the record and its prediction findings.
          </p>

          {deleteError && (
            <Alert tone="danger" title="Could not delete analysis">
              {deleteError}
            </Alert>
          )}

          <div className="flex items-center justify-end gap-3 border-t border-line pt-4">
            <Button
              variant="secondary"
              onClick={() => {
                setItemToDelete(null);
                setDeleteError(null);
              }}
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={handleConfirmDelete}
              loading={deleting}
            >
              Delete analysis
            </Button>
          </div>
        </div>
      </Dialog>
    </PageContainer>
  );
}

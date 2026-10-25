export interface Paginated<T> {
  items: T[];
  total: number | null;
  page: number;
  pageSize: number;
  /** True when another page can be requested. */
  hasMore: boolean;
}

export type SortField = 'created_at' | 'filename' | 'status';
export type SortOrder = 'asc' | 'desc';

export interface ListAnalysesParams {
  page: number;
  pageSize: number;
  query?: string;
  sortBy: SortField;
  order: SortOrder;
}

export interface HealthStatus {
  ok: boolean;
  status?: string;
  detail?: string;
}

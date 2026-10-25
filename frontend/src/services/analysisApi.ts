import { env } from '@/config/env';
import { sleep } from '@/lib/async';
import { asRecord, asString, pick } from '@/lib/safe';
import type { HealthStatus, ListAnalysesParams, Paginated } from '@/types/api';
import { hasPredictions, type AnalysisReport, type AnalysisSummary } from '@/types/analysis';
import { apiClient } from './apiClient';
import { ApiError, defaultMessageFor, isApiError } from './errors';
import { UPLOAD_FIELD_NAME, endpoints } from './endpoints';
import { normalizeList, normalizeReport } from './normalizers';
import type { AnalysisApi, CreateAnalysisOptions } from './types';

const HEALTH_TIMEOUT_MS = 5_000;
const POLL_INTERVAL_MS = 1_500;

async function healthLive(signal?: AbortSignal): Promise<HealthStatus> {
  await apiClient.request(endpoints.healthLive, {
    signal,
    timeoutMs: HEALTH_TIMEOUT_MS,
    suppressUnauthorizedHandler: true,
  });
  return { ok: true, status: 'live' };
}

async function healthReady(signal?: AbortSignal): Promise<HealthStatus> {
  try {
    const raw = await apiClient.request(endpoints.healthReady, {
      signal,
      timeoutMs: HEALTH_TIMEOUT_MS,
      suppressUnauthorizedHandler: true,
    });
    const record = asRecord(raw);
    return {
      ok: true,
      status: asString(pick(record, 'status')) ?? 'ready',
      detail: asString(pick(record, 'detail', 'message')),
    };
  } catch (error) {
    // 503 is the conventional "alive but not ready" answer (e.g. model still loading).
    if (isApiError(error) && error.status === 503) {
      return { ok: false, status: 'not_ready', detail: error.message };
    }
    throw error;
  }
}

async function get(analysisId: string, signal?: AbortSignal): Promise<AnalysisReport> {
  const raw = await apiClient.request(endpoints.analysis(analysisId), { signal });
  return normalizeReport(raw);
}

async function getReport(analysisId: string, signal?: AbortSignal): Promise<AnalysisReport> {
  const raw = await apiClient.request(endpoints.report(analysisId), { signal });
  return normalizeReport(raw);
}

async function getArtifact(
  analysisId: string,
  artifactName: string,
  signal?: AbortSignal,
): Promise<Blob> {
  return apiClient.request<Blob>(endpoints.artifact(analysisId, artifactName), {
    signal,
    responseType: 'blob',
    timeoutMs: env.analysisTimeoutMs,
  });
}

async function list(
  params: ListAnalysesParams,
  signal?: AbortSignal,
): Promise<Paginated<AnalysisSummary>> {
  const raw = await apiClient.request(endpoints.analyses, {
    signal,
    query: {
      page: params.page,
      page_size: params.pageSize,
      q: params.query?.trim() || undefined,
      sort_by: params.sortBy,
      order: params.order,
    },
  });
  return normalizeList(raw, { page: params.page, pageSize: params.pageSize });
}

/** Polls an analysis that the backend accepted but has not finished. */
async function waitForCompletion(analysisId: string, signal?: AbortSignal): Promise<AnalysisReport> {
  const deadline = Date.now() + env.analysisTimeoutMs;
  for (;;) {
    await sleep(POLL_INTERVAL_MS, signal);
    const report = await get(analysisId, signal);
    if (report.status !== 'queued' && report.status !== 'processing') return report;
    if (Date.now() > deadline) {
      throw new ApiError({ kind: 'timeout', message: defaultMessageFor('timeout') });
    }
  }
}

async function create(file: File, options: CreateAnalysisOptions = {}): Promise<AnalysisReport> {
  const form = new FormData();
  form.append(UPLOAD_FIELD_NAME, file, file.name);

  const raw = await apiClient.upload(endpoints.analyses, form, {
    onProgress: options.onUploadProgress,
    signal: options.signal,
    timeoutMs: env.analysisTimeoutMs,
  });

  let report = normalizeReport(raw);
  if (report.status === 'queued' || report.status === 'processing') {
    return waitForCompletion(report.id, options.signal);
  }
  // The upload response may carry only an id and status. Fetch the full result.
  if (report.status !== 'failed' && !hasPredictions(report)) {
    report = await get(report.id, options.signal);
  }
  return report;
}

async function deleteAnalysis(analysisId: string, signal?: AbortSignal): Promise<void> {
  await apiClient.request(endpoints.analysis(analysisId), {
    method: 'DELETE',
    signal,
  });
}

export const analysisApi: AnalysisApi = {
  healthLive,
  healthReady,
  create,
  list,
  get,
  getReport,
  getArtifact,
  delete: deleteAnalysis,
};

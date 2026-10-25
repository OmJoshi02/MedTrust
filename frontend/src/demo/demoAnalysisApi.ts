import { sleep } from '@/lib/async';
import { ApiError } from '@/services/errors';
import type { AnalysisApi, CreateAnalysisOptions } from '@/services/types';
import type { HealthStatus, ListAnalysesParams, Paginated } from '@/types/api';
import {
  DISEASES,
  formatDiseaseName,
  type AnalysisReport,
  type AnalysisSummary,
  type DiseaseName,
} from '@/types/analysis';
import { buildSampleReport, SAMPLE_REPORTS, summarizeReport } from './fixtures';
import { hashString } from './seed';
import { sampleHeatmapSvg, sampleRadiographSvg, svgBlob } from './sampleImages';
import { demoAnalyses, demoSession, demoUploads } from './demoStore';

/** Demo implementation of the analysis API. Every record it returns has `isDemo: true`. */

const HEALTH: HealthStatus = { ok: true, status: 'demo' };

function allReports(): AnalysisReport[] {
  return [...demoAnalyses.all(), ...SAMPLE_REPORTS];
}

function findReport(analysisId: string): AnalysisReport {
  const report = allReports().find((r) => r.id === analysisId);
  if (!report) {
    throw new ApiError({
      kind: 'not_found',
      status: 404,
      message: 'This sample report does not exist in the current demo session.',
    });
  }
  return report;
}

async function create(file: File, options: CreateAnalysisOptions = {}): Promise<AnalysisReport> {
  const { onUploadProgress, signal } = options;

  for (let percent = 0; percent <= 100; percent += 20) {
    onUploadProgress?.(percent);
    await sleep(160, signal);
  }
  await sleep(1800, signal);

  const seed = hashString(`${file.name}:${file.size}`);
  const id = `demo-${Date.now().toString(36)}`;
  const report = buildSampleReport({
    id,
    filename: file.name,
    createdAt: new Date().toISOString(),
    seed,
  });
  report.image = {
    ...report.image,
    sizeBytes: file.size,
    width: undefined,
    height: undefined,
    format: file.type || report.image.format,
    extra: [{ label: 'Source', value: 'Your uploaded file (stays in this browser in demo mode)' }],
  };

  demoUploads.set(id, file);
  demoAnalyses.add(report);
  return report;
}

async function list(
  params: ListAnalysesParams,
  signal?: AbortSignal,
): Promise<Paginated<AnalysisSummary>> {
  await sleep(350, signal);
  if (!demoSession.get()) {
    throw new ApiError({ kind: 'unauthorized', status: 401, message: 'Sign in to view analyses.' });
  }

  const needle = params.query?.trim().toLowerCase() ?? '';
  let items = allReports().map(summarizeReport);
  if (needle !== '') {
    items = items.filter(
      (item) =>
        (item.filename ?? '').toLowerCase().includes(needle) ||
        item.id.toLowerCase().includes(needle) ||
        (item.topFinding ? formatDiseaseName(item.topFinding.disease).toLowerCase().includes(needle) : false),
    );
  }

  const direction = params.order === 'asc' ? 1 : -1;
  items.sort((a, b) => {
    switch (params.sortBy) {
      case 'filename':
        return direction * (a.filename ?? '').localeCompare(b.filename ?? '');
      case 'status':
        return direction * a.status.localeCompare(b.status);
      default:
        return direction * ((a.createdAt ?? '') < (b.createdAt ?? '') ? -1 : 1);
    }
  });

  const total = items.length;
  const start = (params.page - 1) * params.pageSize;
  return {
    items: items.slice(start, start + params.pageSize),
    total,
    page: params.page,
    pageSize: params.pageSize,
    hasMore: start + params.pageSize < total,
  };
}

async function get(analysisId: string, signal?: AbortSignal): Promise<AnalysisReport> {
  await sleep(250, signal);
  return findReport(analysisId);
}

async function getArtifact(
  analysisId: string,
  artifactName: string,
  signal?: AbortSignal,
): Promise<Blob> {
  await sleep(150, signal);
  const report = findReport(analysisId);
  const seed = hashString(report.id);

  if (artifactName === 'original') {
    return demoUploads.get(report.id) ?? svgBlob(sampleRadiographSvg(seed));
  }
  if (artifactName.startsWith('gradcam_')) {
    const key = artifactName.slice('gradcam_'.length);
    const disease: DiseaseName | undefined = DISEASES.find((d) => d.toLowerCase() === key);
    const label = disease ? formatDiseaseName(disease) : 'Region of interest';
    return svgBlob(sampleHeatmapSvg(seed, disease, label));
  }
  throw new ApiError({ kind: 'not_found', status: 404, message: 'Artifact not found.' });
}

async function deleteAnalysis(analysisId: string, signal?: AbortSignal): Promise<void> {
  await sleep(150, signal);
  demoAnalyses.delete(analysisId);
  demoUploads.delete(analysisId);
}

export const demoAnalysisApi: AnalysisApi = {
  healthLive: async () => HEALTH,
  healthReady: async () => HEALTH,
  create,
  list,
  get,
  getReport: get,
  getArtifact,
  delete: deleteAnalysis,
};

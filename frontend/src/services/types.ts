import type { HealthStatus, ListAnalysesParams, Paginated } from '@/types/api';
import type { AnalysisReport, AnalysisSummary } from '@/types/analysis';
import type { LoginCredentials, RegisterPayload, RegisterResult, User } from '@/types/auth';

export interface AuthApi {
  register(payload: RegisterPayload): Promise<RegisterResult>;
  login(credentials: LoginCredentials): Promise<User>;
  logout(): Promise<void>;
  /** Resolves with the current user, or rejects with an "unauthorized" ApiError. */
  me(signal?: AbortSignal): Promise<User>;
}

export interface CreateAnalysisOptions {
  /** Upload progress, 0–100. */
  onUploadProgress?: (percent: number) => void;
  signal?: AbortSignal;
}

export interface AnalysisApi {
  healthLive(signal?: AbortSignal): Promise<HealthStatus>;
  healthReady(signal?: AbortSignal): Promise<HealthStatus>;
  /** Uploads the file and resolves with the finished analysis. */
  create(file: File, options?: CreateAnalysisOptions): Promise<AnalysisReport>;
  list(params: ListAnalysesParams, signal?: AbortSignal): Promise<Paginated<AnalysisSummary>>;
  get(analysisId: string, signal?: AbortSignal): Promise<AnalysisReport>;
  getReport(analysisId: string, signal?: AbortSignal): Promise<AnalysisReport>;
  getArtifact(analysisId: string, artifactName: string, signal?: AbortSignal): Promise<Blob>;
  delete(analysisId: string, signal?: AbortSignal): Promise<void>;
}

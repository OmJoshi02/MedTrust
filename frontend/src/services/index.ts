import { env } from '@/config/env';
import { demoAnalysisApi, demoAuthApi } from '@/demo';
import { analysisApi } from './analysisApi';
import { authApi } from './authApi';
import type { AnalysisApi, AuthApi } from './types';

/**
 * The single place where the real API and the demo implementation are chosen.
 * UI code imports `authService` / `analysisService` and never knows which one
 * it received. The real modules never import anything from `src/demo`.
 */
export const authService: AuthApi = env.demoMode ? demoAuthApi : authApi;
export const analysisService: AnalysisApi = env.demoMode ? demoAnalysisApi : analysisApi;

export { apiClient } from './apiClient';
export { ApiError, describeError, isApiError } from './errors';
export type { ApiErrorKind } from './errors';
export type { AnalysisApi, AuthApi, CreateAnalysisOptions } from './types';

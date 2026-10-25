import { env } from '@/config/env';

/**
 * Every backend path in one place. Adapt here if the backend differs.
 * The prefix is configurable with VITE_API_PREFIX.
 */
const p = env.apiPrefix;
const seg = (value: string) => encodeURIComponent(value);

export const endpoints = {
  healthLive: `${p}/health/live`,
  healthReady: `${p}/health/ready`,

  register: `${p}/auth/register`,
  login: `${p}/auth/login`,
  logout: `${p}/auth/logout`,
  me: `${p}/auth/me`,

  analyses: `${p}/analyses`,
  analysis: (analysisId: string) => `${p}/analyses/${seg(analysisId)}`,
  report: (analysisId: string) => `${p}/analyses/${seg(analysisId)}/report`,
  artifact: (analysisId: string, artifactName: string) =>
    `${p}/analyses/${seg(analysisId)}/artifacts/${seg(artifactName)}`,
} as const;

/** Multipart field name used for the uploaded X-ray in POST /analyses. */
export const UPLOAD_FIELD_NAME = 'file';

/** Artifact name used for the original image unless the report says otherwise. */
export const ORIGINAL_ARTIFACT_NAME = 'original';

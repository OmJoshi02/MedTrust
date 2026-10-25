/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_API_PREFIX?: string;
  readonly VITE_DEMO_MODE?: string;
  readonly VITE_REQUEST_TIMEOUT_MS?: string;
  readonly VITE_ANALYSIS_TIMEOUT_MS?: string;
  readonly VITE_MAX_UPLOAD_MB?: string;
  readonly VITE_ENABLE_REMEMBER_ME?: string;
  readonly VITE_INSTITUTION?: string;
  readonly VITE_CONTACT_EMAIL?: string;
  readonly VITE_PROJECT_TEAM?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ANALYSIS_MODE?: 'demo' | 'api';
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

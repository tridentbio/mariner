/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WS_URL: string;
  readonly VITE_API_BASE_URL: string;
  readonly VITE_MOCK_API: 0 | 1;
}

interface ImportMeta {
  readonly env: ImporteMtaEnv;
}

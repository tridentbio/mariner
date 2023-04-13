/// <reference types="vite/client" />
//
// Configurations to build and run the development server
//
// By default, performs code splitting to allow a faster UX.
//
// This behavior breaks e2e tests and may be unwanted. To disable it
// run the vite commands with SKIP_CODE_SPLITTING=true environment variable
//
//    SKIP_CODE_SPLITTING=true npx vite
//    SKIP_CODE_SPLITTING=true npm start status

import { defineConfig, UserConfig } from 'vite';
import react from '@vitejs/plugin-react';


let config: UserConfig = {
  plugins: [],
  build: {
  },
};


export default defineConfig(config);

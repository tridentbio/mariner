/// <reference types="vite/client" />
// Configurations to build and run the development server
//
// By default, performs code splitting to allow a faster UX.
//
// This behavior breaks e2e tests and may be unwanted. To disable it
// run the vite commands with SKIP_CODE_SPLITTING=true environment variable
//
//    SKIP_CODE_SPLITTING=true npx vite
//    SKIP_CODE_SPLITTING=true npm start status

import { defineConfig, } from 'vite';
import react from '@vitejs/plugin-react';
import viteTsconfigPaths from 'vite-tsconfig-paths';
import svgrPlugin from 'vite-plugin-svgr';

/** @type {import('vite').UserConfig} */
let config = {
  plugins: [react(), viteTsconfigPaths(), svgrPlugin()],
  build: {
    outDir: 'build',
  },
  /**
   * Avoids Cypress E2E tests error caused by PopperJS dependencies with Material UI
   * @link https://github.com/vitejs/vite/issues/12423
   */
  optimizeDeps: {
    include: ['@mui/material/Tooltip'],
  },
  server: {
    host: true,
    open: false,
    port: 3000,
  },
};

// Function to fix build config when running through cypress
function removeCodeSplittingConfigProps() {
  const isCodeSplitting = !process.env.SKIP_CODE_SPLITTING;
  // Removes properties that break on cypress
  if (!isCodeSplitting && config.build?.rollupOptions) {
    console.log('Not using code splitting!');
    config.build.rollupOptions = undefined;
    delete config.build['rollupOptions'];
  }
}

removeCodeSplittingConfigProps();

export default defineConfig(config);

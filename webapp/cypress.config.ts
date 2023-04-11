import { defineConfig } from 'cypress';
import vitePreprocessor from 'cypress-vite';

export default defineConfig({
  env: {},
  e2e: {
    baseUrl: 'http://localhost:3000/',
    setupNodeEvents(on) {
      on('file:preprocessor', vitePreprocessor());
    },
  },
});

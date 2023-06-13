const vitePreprocessor = require('cypress-vite');
const { defineConfig } = require('cypress');

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost';
const DATA_PATH = process.env.DATA_PATH || 'cypress/fixtures/data';

module.exports = defineConfig({
  env: {
    API_BASE_URL: API_BASE_URL,
    DATA_PATH: DATA_PATH,
  },
  e2e: {
    baseUrl: 'http://localhost:3000/',
    setupNodeEvents(on) {
      on('file:preprocessor', vitePreprocessor());
    },
  },
});

const vitePreprocessor = require("cypress-vite");
const { defineConfig } = require("cypress");

const API_BASE_URL = process.env.VITE_API_BASE_URL || 'http://localhost:8000';
const DATA_PATH = process.env.DATA_PATH || 'cypress/fixtures/data';
const SCHEMA_PATH = process.env.SCHEMA_PATH || '../backend/tests/data';
const TEST_USER = 'test@domain.com'

module.exports = defineConfig({
  env: {
    API_BASE_URL: API_BASE_URL,
    DATA_PATH: DATA_PATH,
    TEST_USER: TEST_USER,
    SCHEMA_PATH: SCHEMA_PATH,
  },

  e2e: {
    baseUrl: "http://localhost:3000/",
    setupNodeEvents(on) {
      on("file:preprocessor", vitePreprocessor());
    },
  },

  component: {
    viewportHeight: 800,
    viewportWidth: 1200,
    devServer: {
      framework: "react",
      bundler: "vite",
    },
  },
});

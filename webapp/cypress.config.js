const vitePreprocessor = require("cypress-vite");
const { defineConfig } = require("cypress");

module.exports = defineConfig({
  env: {
    TEST_USER: 'test@domain.com',
    SCHEMA_PATH: `${__dirname}/../backend/tests/data`,
    API_BASE_URL: 'http://localhost:8000', 
  },

  e2e: {
    baseUrl: "http://localhost:3000/",
    setupNodeEvents(on) {
      on("file:preprocessor", vitePreprocessor());
    },
    video: true,
  },

  component: {
    viewportHeight: 800,
    viewportWidth: 1200,
    video: true,
    devServer: {
      framework: "react",
      bundler: "vite",
    },
  },
});

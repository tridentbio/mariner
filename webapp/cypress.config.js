const vitePreprocessor = require("cypress-vite");
const { defineConfig } = require("cypress");

module.exports = defineConfig({
  env: {
    TEST_USER: {
      username: 'test@domain.com',
      password: '123456'
    },
    ADMIN_USER: {
      username: 'admin@mariner.trident.bio',
      password: '123456',
    },
    SCHEMA_PATH: `${__dirname}/../backend/tests/data`,
    API_BASE_URL: 'http://localhost:8000', 
  },

  e2e: {
    viewportHeight: 768,
    viewportWidth: 1366,
    baseUrl: "http://localhost:3000/",
    setupNodeEvents(on) {
      on("file:preprocessor", vitePreprocessor());
    },
    video: true,
    videoCompression: 32,
    /** Keeps user login session across each test */
    testIsolation: false
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

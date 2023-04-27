const vitePreprocessor = require('cypress-vite')
const { defineConfig } = require('cypress')

module.exports = defineConfig({
  env: {},
  e2e: {
    baseUrl: 'http://localhost:3000/',
    setupNodeEvents(on) {
      on('file:preprocessor', vitePreprocessor());
    },
  },
});

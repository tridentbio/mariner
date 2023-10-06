import vitePreprocessor from "cypress-vite"
import { defineConfig } from "cypress"
import cypressSplit from 'cypress-split'
import cypressOnFix from 'cypress-on-fix'
import fs from 'fs'

const preventSuccessfullTestsVideoGeneration = (spec: Cypress.Spec, results: CypressCommandLine.RunResult) => {
  if (results && results.video) {
    // Do we have failures for any retry attempts?
    const failures = results.tests.some((test) =>
      test.attempts.some((attempt) => attempt.state === 'failed')
    )
    if (!failures) {
      // delete the video if the spec passed and no tests retried
      fs.unlinkSync(results.video)
    }
  }
}

export default defineConfig({
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
    setupNodeEvents(cypressOn, config) {
      //? Fix bug on Cypress not handling multiple listeners for the same event
      //? Reason: `cypressSplit` overwrites some of them
      const on: Cypress.PluginEvents = cypressOnFix(cypressOn)
      
      on("file:preprocessor", vitePreprocessor());
      on(
        'after:spec',
        preventSuccessfullTestsVideoGeneration
      );
      
      cypressSplit(on, config)

      //! IMPORTANT: return the config object (required for cypress split)
      return config
    },
    video: true,
    videoCompression: 32,
    /** Keeps user login session across each test */
    testIsolation: false
  },

  component: {
    viewportHeight: 768,
    viewportWidth: 1366,
    video: true,
    devServer: {
      framework: "react",
      bundler: "vite",
    },
    setupNodeEvents(on) {
      on(
        'after:spec',
        preventSuccessfullTestsVideoGeneration
      )
    },
  },
});

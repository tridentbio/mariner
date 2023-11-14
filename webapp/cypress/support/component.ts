/* eslint-disable no-console */
// ***********************************************************
// This example support/component.js is processed and
// loaded automatically before your test files.
//
// This is a great place to put global configuration and
// behavior that modifies Cypress.
//
// You can change the location of this file or turn off
// automatically serving support files with the
// 'supportFile' configuration option.
//
// You can read more here:
// https://on.cypress.io/configuration
// ***********************************************************

// Import commands.js using ES2015 syntax:
// import './commands'

// Alternatively you can use CommonJS syntax:
// require('./commands')

import { mount } from 'cypress/react'
import { fakeApi } from '../../src/mock/msw/server'
import { rest } from 'msw'
import '@4tw/cypress-drag-drop'
import 'cypress-file-upload'

Cypress.Commands.add('mount', mount)

Cypress.Commands.add('notificationShouldContain', (text: string) => {
  return cy
    .get('.MuiAlert-message', { timeout: 30000 })
    .should('contain.text', text);
});

declare global {
  interface Window {
    msw?: {
      fakeApi: typeof fakeApi
      rest: typeof rest
    }
  }
}

Cypress.on('test:before:run:async', async () => {
  if(window.msw) {
    console.log('MSW is already running.')
  } else {
    console.log('MSW has not been started. Starting now.')

    window.msw = { fakeApi, rest }

    await fakeApi.start();
  }
});
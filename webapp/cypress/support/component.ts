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
import '@4tw/cypress-drag-drop'
import { fakeApi } from 'mock/msw/server'

Cypress.Commands.add('mount', mount)

before(() => fakeApi.start({  onUnhandledRequest: 'bypass' }))

  //? Reset handlers so that each test could alter them
  //? without affecting other, unrelated tests.
  afterEach(() => fakeApi.resetHandlers())

  after(() => fakeApi.stop())

// Example use:
// cy.mount(<MyComponent />)
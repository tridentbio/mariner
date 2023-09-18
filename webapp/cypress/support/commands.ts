/// <reference types="cypress" />

import 'cypress-plugin-tab';
import 'cypress-file-upload';
import './dataset';
import './models';
import './deployments';
import { drag, move } from './dragdrop';
import { deleteDatasetIfAlreadyExists } from './dataset/delete';
import { mount } from 'cypress/react';

const TEST_USER = Cypress.env('TEST_USER');

Cypress.Commands.add('notificationShouldContain', (text: string) => {
  return cy
    .get('.MuiAlert-message', { timeout: 20000 })
    .should('contain.text', text);
});

Cypress.Commands.add('loginSuper', (timeout: number = 15000) => {
  cy.clearAllCookies();
  cy.visit('/login');
  cy.get('#username-input', { timeout }).type('admin@mariner.trident.bio');
  cy.get('#password-input').type('123456');
  cy.get('button[type="submit"]').click();
  cy.url().should('eq', Cypress.config('baseUrl'));
});

Cypress.Commands.add('loginTest', (timeout: number = 15000) => {
  cy.clearAllCookies();
  cy.visit('/login').then(() => {
    cy.get('#username-input', { timeout }).type(TEST_USER);
    cy.get('#password-input').type('123456');
    cy.get('button[type="submit"]').click();
    cy.url().should('eq', Cypress.config('baseUrl'));
  });
});

export const addDescription = (
  pattern: string,
  dataType: {
    domainKind: 'Numeric' | 'SMILES' | 'Categorical' | 'String';
    unit?: string;
  },
  description = ''
) => {
  cy.get('[data-testid="dataset-col-name-input"]')
    .get(`input[value="${pattern}"]`)
    .first();
  cy.get(`[data-testid="input-group-${pattern}"]`)
    .get(`[data-testid="data-type-input-${pattern}"]`)
    .last()
    .click()
    .get('li[role="option"]')
    .contains(dataType.domainKind)
    .click();
  if (dataType.unit)
    cy.get(`[data-testid="input-group-${pattern}"]`)
      .get(`[data-testid="dataset-col-data-type-unit-${pattern}"]`)
      .last()
      .click()
      .wait(300)
      .type(dataType.unit)
      .wait(800)
      .get('li[role="option"]')
      .contains(dataType.unit)
      .last()
      .click();
  cy.get('[data-testid="dataset-col-description-input"]')
    .get(`[data-descriptionPattern=${pattern}]`)
    .first()
    .click()
    .type(description);
};

Cypress.Commands.add('deleteZINCDataset', () => {
  deleteDatasetIfAlreadyExists('Some dataset');
});

Cypress.Commands.add(
  'drag',
  {
    prevSubject: 'element',
  },
  (draggedElement, dropSelector, dropX = 0, dropY = 0) => {
    drag(draggedElement, dropSelector, dropX, dropY);
  }
);

Cypress.Commands.add(
  'move',
  {
    prevSubject: 'element',
  },
  (draggedJquery, dropSelector, dropX = 0, dropY = 0) => {
    move(draggedJquery, dropSelector, dropX, dropY);
  }
);

Cypress.Commands.add('getWithoutThrow', (selector: string) => {
  cy.get('body').then(($body) => {
    if ($body.find(selector).length) {
      return cy.get(selector);
    } else {
      return [];
    }
  });
});

Cypress.Commands.add('getCurrentAuthString', () =>
  cy.window().then((win) => {
    const value = JSON.parse(win.localStorage.getItem('app-token') || '{}');
    if (!value.access_token) {
      throw new Error('No access token found');
    }
    return cy.wrap(`Bearer ${value.access_token}`);
  })
);

declare global {
  namespace Cypress {
    interface Chainable {
      loginSuper(timeout?: number): Chainable<void>;
      loginTest(timeout?: number): Chainable<void>;
      createZINCDataset(): Chainable<void>;
      deleteZINCDataset(): Chainable<void>;
      notificationShouldContain(text: string): Chainable<JQuery<HTMLElement>>;
      move(
        dropSelector: string,
        x: number,
        y: number
      ): Chainable<JQuery<HTMLElement>>;
      drag(
        dropSelector: string,
        x: number,
        y: number
      ): Chainable<JQuery<HTMLElement>>;
      getWithoutThrow(selector: string): Chainable<JQuery<HTMLElement>>;
      getCurrentAuthString(): Chainable<string>;
      mount: typeof mount;
    }
  }
}

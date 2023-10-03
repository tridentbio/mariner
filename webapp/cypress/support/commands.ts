/// <reference types="cypress" />

import 'cypress-file-upload';
import './dataset';
import './models';
import './deployments';
import { deleteDatasetIfAlreadyExists } from './dataset/delete';
import { mount } from 'cypress/react';
import '@4tw/cypress-drag-drop'
import { drag, move } from './custom-dragdrop';
import { ELocalStorage, fetchLocalStorage } from '@app/local-storage';

const TEST_USER = Cypress.env('TEST_USER');
const ADMIN_USER = Cypress.env('ADMIN_USER');

Cypress.Commands.add('notificationShouldContain', (text: string) => {
  return cy
    .get('.MuiAlert-message', { timeout: 30000 })
    .should('contain.text', text);
});

Cypress.Commands.add('loginSuper', (role = 'admin', timeout: number = 15000) => {
  const loginData = (() => {
    switch(role) {
      case 'test': return {username: TEST_USER.username, password: TEST_USER.password}
      default: return {username: ADMIN_USER.username, password: ADMIN_USER.password}
    }
  })()

  cy.session([loginData.username, loginData.password], () => {
    cy.visit('/login');
    cy.get('#username-input', { timeout }).type(loginData.username);
    cy.get('#password-input').type(loginData.password);
    cy.get('button[type="submit"]').click();

    cy.url().should('eq', Cypress.config('baseUrl'))
      .then(() => {
        const storage = fetchLocalStorage(ELocalStorage.TOKEN)
    
        expect(storage).to.have.property('access_token');
      });

  }, {cacheAcrossSpecs: true});
});

Cypress.Commands.add(
  'customDrag',
  {
    prevSubject: 'element',
  },
  (draggedElement, dropSelector, dropX = 0, dropY = 0) => {
    drag(draggedElement, dropSelector, dropX, dropY);
  }
);

Cypress.Commands.add(
  'customMove',
  {
    prevSubject: 'element',
  },
  (draggedJquery, dropSelector, dropX = 0, dropY = 0) => {
    move(draggedJquery, dropSelector, dropX, dropY);
  }
);

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
      loginSuper(role?: 'admin' | 'test', timeout?: number): Chainable<void>;
      createZINCDataset(): Chainable<void>;
      deleteZINCDataset(): Chainable<void>;
      notificationShouldContain(text: string): Chainable<JQuery<HTMLElement>>;
      getWithoutThrow(selector: string): Chainable<JQuery<HTMLElement>>;
      getCurrentAuthString(): Chainable<string>;
      mount: typeof mount;
      customMove(
        dropSelector: string,
        x: number,
        y: number
      ): Chainable<JQuery<HTMLElement>>;
      customDrag(
        dropSelector: string,
        x: number,
        y: number
      ): Chainable<JQuery<HTMLElement>>;
    }
  }
}

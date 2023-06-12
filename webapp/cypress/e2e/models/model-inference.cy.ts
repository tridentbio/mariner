import { Model } from '@app/types/domain/models';
import { trainModel } from '../../support/training/create';

const API_BASE_URL = Cypress.env('API_BASE_URL');

describe('/models/:modelId/inference', () => {
  let modelName: string | null = null;

  beforeEach(() => {
    cy.loginSuper();
    cy.setupSomeModel().then((name) => {
      modelName = name;
    });
  });

  it('Visits the page and inference ', () => {
    cy.intercept({
      method: 'GET',
      url: `${API_BASE_URL}/api/v1/models/?page=0&perPage=10`,
    }).as('getModels');
    cy.intercept({
      method: 'GET',
      url: `${API_BASE_URL}/api/v1/models/*`,
    }).as('getSpecificModel');
    cy.intercept({
      method: 'GET',
      url: `${API_BASE_URL}/api/v1/experiments/*`,
    }).as('getExperiments');

    cy.then(() =>
      trainModel(modelName!, {
        batchSize: 8,
        learningRate: 0.001,
      })
    );

    cy.visit('/models');
    cy.wait('@getModels');
    cy.contains('a', modelName!).click();
    cy.wait('@getSpecificModel').wait(500);
    cy.wait('@getExperiments').wait(500);
    cy.get('button').contains('Inference').click().wait(300);
    cy.get('#model-version-select').click();
    cy.get('ul[role="listbox"]').within(() => {
      cy.get('li').first().click().wait(1000);
    });
    cy.contains('div', 'Inputs:')
      .find('input')
      .each(($el) => {
        cy.wrap($el).type(Math.random().toPrecision(3));
      });
    cy.get('button').contains('Predict').click().wait(5000);
    cy.get('[data-testid="inference-result"]').should('exist');
  });
});

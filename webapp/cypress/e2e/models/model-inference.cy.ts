import { Model } from 'app/types/domain/models';

describe('/models/:modelId/inference', () => {
  beforeEach(() => {
    cy.loginSuper();
  });

  it('Visits the page and inference ', () => {
    cy.intercept({
      method: 'GET',
      url: 'http://localhost/api/v1/models/?page=0&perPage=10',
    }).as('getModels');
    cy.intercept({
      method: 'GET',
      url: 'http://localhost/api/v1/models/*',
    }).as('getSpecificModel');
    cy.intercept({
      method: 'GET',
      url: 'http://localhost/api/v1/experiments/*',
    }).as('getExperiments');
    cy.visit('/models');
    cy.wait('@getModels');
    cy.get('tbody').find('a').first().click();
    cy.wait('@getSpecificModel').wait(500);
    cy.wait('@getExperiments').wait(500);
    cy.get('button').contains('Inference').click().wait(300);
    cy.get('#model-version-select').click();
    cy.get('ul[role="listbox"]').within(() => {
      cy.get('li').first().click().wait(1000);
    });
    cy.get('span').contains('JSME Input').should('exist');
    cy.get('div[role="graphics-document"]').should('exist');
    cy.get('span').contains('Textbox Input').click().wait(250);
    cy.get('[data-testid="input-textbox"]').click().type('CCCC');
    cy.get('#categories-label')
      .parent()
      .click()
      .get('li[role="option"]')
      .first()
      .click()
      .wait(150);
    cy.get('button').contains('Predict').click().wait(5000);
    cy.get('[data-testid="inference-result"]').should('exist');
  });
});

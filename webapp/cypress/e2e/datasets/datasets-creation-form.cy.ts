import { zincDatasetFixture } from '../../support/dataset/examples';
import TestUtils from '../../support/TestUtils';

describe('/datasets/new - Dataset form', () => {
  beforeEach(() => {
    cy.loginSuper();
    cy.visit('/datasets');
    cy.intercept({
      method: 'GET',
      url: '/api/v1/datasets/?*',
    }).as('getDatasets');
    cy.wait('@getDatasets').then(({ response }) => {
      expect(response?.statusCode).to.eq(200);
    });
    cy.get('button', { timeout: 2000 }).contains('Add Dataset').click();
  });

  it('Feedback missing required fields', () => {
    cy.get('button').contains('Save').click();
    cy.wait(300);
    cy.get('#dataset-name-input')
      .parent({ timeout: 3000 })
      .should('have.class', TestUtils.errorClass);
    cy.get('#description-input')
      .siblings()
      .should('have.class', TestUtils.errorClass);
    cy.get('#dataset-upload')
      .siblings()
      .should('have.class', TestUtils.errorClass);

    cy.intercept({
      method: 'POST',
      url: '/api/v1/datasets/csv-metadata',
    }).as('getCsvData');
    cy.get('#dataset-upload').attachFile(zincDatasetFixture.file).wait(2000);

    cy.wait('@getCsvData').then(({ response }) => {
      expect(response?.statusCode).to.eq(200);
    });

    cy.get('button').contains('Save').click().wait(300);

    cy.get('#dataset-name-input')
      .parent()
      .should('have.class', TestUtils.errorClass);
    cy.get('#description-input')
      .siblings()
      .should('have.class', TestUtils.errorClass);

    cy.get('#dataset-split-input')
      .parent()
      .should('have.class', TestUtils.errorClass);
  });

  it('Shows required split column when split type is not random', () => {
    cy.get('#dataset-upload').attachFile(zincDatasetFixture.file);
    cy.get('#dataset-splittype-input', { timeout: 60000 })
      .click()
      .get('li')
      .contains('Scaffold')
      .click();
    cy.get('#dataset-split-column-input').should('exist');
    cy.get('button').contains('Save').click();
    cy.get('#dataset-split-column-input', { timeout: 2000 })
      .parent()
      .should('have.class', TestUtils.errorClass);
  });
});

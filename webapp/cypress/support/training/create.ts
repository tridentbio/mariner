import { randomLowerCase } from 'utils';

export const checkModelTraining = (modelName?: string) => {
  cy.once('uncaught:exception', () => false);
  // Visits models listing page
  cy.intercept({
    url: 'http://localhost/api/v1/experiments',
    method: 'POST',
  }).as('createExperiment');
  cy.intercept({
    url: 'http://localhost/api/v1/models/*',
    method: 'GET',
  }).as('getModels');
  cy.visit('/models');
  cy.wait('@getModels').then(({ response }) => {
    expect(response?.statusCode).to.eq(200);
    cy.log(response?.body.data[0].versions[0].id);
  });
  if (modelName) {
    cy.get('table a').contains(modelName).click();
  } else {
    cy.get('table a').first().click();
  }
  cy.get('div[role="tablist"] button').contains('Training').click().wait(1000);
  cy.get('button').contains('Create Training').click();
  // Fills the experiment form
  const experimentName = randomLowerCase(8);
  cy.get('[aria-label="experiment name"] input').type(experimentName);
  cy.get('#model-version-select')
    .click()
    .get('li[role="option"]')
    .first()
    .click();
  cy.contains('div', 'Experiment Name').find('input').type(randomLowerCase(8));
  cy.contains('div', 'Model Version').find('input').click();
  cy.get('li[role="option"]').first().click();
  cy.contains('div', 'Learning Rate').find('input').clear().type('0.05');
  cy.contains('div', 'Batch Size').find('input').clear().type('32');
  cy.contains('div', 'Epochs').find('input').clear().type('10');
  cy.contains('div', 'Metric to monitor').click();
  cy.get('li[role="option"]').first().click();
  cy.contains('div', 'Target Column').click();
  cy.get('li[role="option"]').first().click();
  // Selects CREATE EXPERIMENT
  cy.get('button').contains('CREATE').click();
  // Assert API call is successfull
  cy.wait('@createExperiment', { timeout: 30000 }).then(({ response }) => {
    expect(response?.statusCode).to.eq(200);
  });
};

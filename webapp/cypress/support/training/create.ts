import { randomLowerCase } from 'utils';

export const checkModelTraining = (modelName?: string) => {
  // Visits models listing page
  cy.intercept(
    {
      url: 'http://localhost/api/v1/experiments',
      method: 'POST',
    },
    { statusCode: 200 }
  ).as('createExperiment');
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
  cy.get('[aria-label="learning rate"]').type('0.05');
  cy.get('[aria-label="batch size"]').type('32');
  cy.get('[aria-label="max epochs"]').type(modelName ? '10' : '500');
  // Selects CREATE EXPERIMENT
  cy.get('button').contains('CREATE').click();
  // Assert API call is successfull
  cy.wait('@createExperiment', { timeout: modelName ? 30000 : 3000 }).then(
    ({ response }) => {
      expect(response?.statusCode).to.eq(200);
    }
  );
};

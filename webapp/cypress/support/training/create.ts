import { randomLowerCase } from 'utils';

type TrainingConfig = {
  batchSize?: string | number;
  epochs?: string | number;
  learningRate?: string | number;
  modelVersion?: string;
};

const defaultTrainingConfig: TrainingConfig = {
  batchSize: 32,
  epochs: 1,
  learningRate: 0.05,
};

const API_BASE_URL = Cypress.env('API_BASE_URL');

const checkTrainFinishes = (
  experimentName: string,
  timeout = 60000
): Cypress.Chainable<boolean> =>
  // Recursively check if the experiment row contains 'Trained' status
  cy.contains('tr', experimentName).then(($row) => {
    if ($row.text().includes('Trained')) return cy.wrap(true);
    else if (timeout <= 0) return cy.wrap(false);
    else
      return cy
        .wait(1000)
        .then(() => checkTrainFinishes(experimentName, timeout - 1000));
  });

export const trainModel = (modelName?: string, config: TrainingConfig = {}) => {
  config = { ...defaultTrainingConfig, ...config };
  cy.once('uncaught:exception', () => false);
  // Visits models listing page
  cy.intercept({
    url: `${API_BASE_URL}/api/v1/experiments`,
    method: 'POST',
  }).as('createExperiment');
  cy.intercept({
    url: `${API_BASE_URL}/api/v1/models/*`,
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

  if (config.modelVersion) {
    cy.get('#model-version-select')
      .click()
      .get('li[role="option"]')
      .contains(config.modelVersion)
      .click();
  } else {
    cy.get('#model-version-select')
      .click()
      .get('li[role="option"]')
      .first()
      .click();
  }

  cy.contains('div', 'Experiment Name').find('input').type(randomLowerCase(8));
  cy.contains('div', 'Model Version').find('input').click();
  cy.get('li[role="option"]').first().click();
  cy.contains('div', 'Learning Rate')
    .find('input')
    .clear()
    .type(config.learningRate?.toString()!);
  cy.contains('div', 'Batch Size')
    .find('input')
    .clear()
    .type(config.batchSize?.toString()!);
  cy.contains('div', 'Epochs')
    .find('input')
    .clear()
    .type(config.epochs?.toString()!);
  cy.contains('div', 'Target Column').click();
  cy.get('li[role="option"]').first().click();
  cy.contains('div', 'Metric to monitor').click();
  cy.get('li[role="option"]').first().click();
  // Selects CREATE EXPERIMENT
  cy.get('button').contains('CREATE').click();
  // Assert API call is successfull
  return cy
    .wait('@createExperiment', { timeout: 10000 })
    .then(({ response }) => {
      expect(response?.statusCode).to.eq(200);
      return cy.wrap(response);
    })
    .then((response) => {
      const experimentName = response?.body.experimentName
      checkTrainFinishes(experimentName).then((trained) =>
        assert.isTrue(trained)
      );
      return cy.wrap(response?.body);
    });
};

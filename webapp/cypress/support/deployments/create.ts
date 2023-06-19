import { randomLowerCase } from '@utils';

const API_BASE_URL = Cypress.env('API_BASE_URL');

const getDeploymentForm = (
  modelVersionName: string,
  shareStrategy: 'Private' | 'Public' = 'Private'
) => ({
  name: randomLowerCase(8),
  readme: '### This is a test deployment',
  shareStrategy,
  modelVersion: modelVersionName,
});

export const goToDeploymentWithinModel = (modelName: string) => {
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/models/?page=0&perPage=10`,
  }).as('getModels');
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/models/*`,
  }).as('getSpecificModel');
  cy.visit('/models');
  cy.wait('@getModels', { timeout: 10000 }).wait(500);
  cy.contains('a', modelName!).click();
  cy.wait('@getSpecificModel').wait(500);
  cy.get('button').contains('Deployments').click().wait(300);
};

const fillForm = (
  deploymentFormData: Partial<ReturnType<typeof getDeploymentForm>>
) => {
  cy.intercept({
    method: 'POST',
    url: `${API_BASE_URL}/api/v1/deployments/`,
  }).as('createDeployment');

  deploymentFormData.name &&
    cy.get('input[name="name"]').type(deploymentFormData.name);

  deploymentFormData.readme &&
    cy.get('textarea').type(deploymentFormData.readme);

  deploymentFormData.shareStrategy &&
    cy
      .contains('div', 'Share Strategy')
      .get('span')
      .contains(deploymentFormData.shareStrategy)
      .click();

  if (deploymentFormData.modelVersion) {
    cy.get('#model-version-select').click();
    cy.get('ul[role="listbox"]').within(() => {
      cy.get('li').contains(deploymentFormData.modelVersion!).click();
    });
  } else {
    cy.get('#model-version-select').click();
    cy.get('ul[role="listbox"]').within(() => {
      cy.get('li').first().click().wait(1000);
    });
  }
};

export const createDeployment = (
  modelName: string,
  modelVersionName: string
) => {
  goToDeploymentWithinModel(modelName);

  cy.get('button').contains('Deploy Model').click().wait(300);
  const deploymentsFormData = getDeploymentForm(modelVersionName);
  fillForm(deploymentsFormData);

  cy.contains('button', 'CREATE').click();
  cy.wait('@createDeployment').then((interception) => {
    assert.equal(interception.response?.statusCode, 200);
  });

  cy.get('td').should('contain', deploymentsFormData.name);
  cy.get('td').should('contain', deploymentsFormData.modelVersion);

  return cy.wrap(deploymentsFormData.name);
};

import { randomLowerCase } from '@utils';
import { goToDeploymentWithinModel } from './find-deployment';

const API_BASE_URL = Cypress.env('API_BASE_URL');

const getDeploymentForm = (
  modelVersionName: string,
  shareStrategy: 'Private' | 'Public' = 'Private'
) => ({
  name: randomLowerCase(8),
  readme: '### This is a test deployment',
  shareStrategy,
  shareWithUser: [] as string[],
  shareWithTeam: [] as string[],
  modelVersion: modelVersionName,
});

const fillForm = (
  deploymentFormData: Partial<ReturnType<typeof getDeploymentForm>>
) => {
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

  if (deploymentFormData.shareWithUser?.length) {
    deploymentFormData.shareWithUser.forEach((user) => {
      cy.get('#tags-outlined').type(user);
      cy.get('ul[role="listbox"]').within(() => {
        cy.get('li').first().click().wait(1000);
      });
    });
  }

  if (deploymentFormData.shareWithTeam?.length) {
    deploymentFormData.shareWithTeam.forEach((team) => {
      cy.get('#share-with-organization-domain').type(team);
      cy.get('button').contains('Add').click();
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

  cy.intercept({
    method: 'POST',
    url: `${API_BASE_URL}/api/v1/deployments/`,
  }).as('createDeployment');
  cy.contains('button', 'CREATE').click();
  cy.wait('@createDeployment').then((interception) => {
    assert.equal(interception.response?.statusCode, 200);
  });

  cy.get('td').should('contain', deploymentsFormData.name);
  cy.get('td').should('contain', deploymentsFormData.modelVersion);

  return cy.wrap(deploymentsFormData.name);
};

export const updateDeployment = (
  modelName: string,
  deploymentName: string,
  deploymentFormData: Partial<ReturnType<typeof getDeploymentForm>>
) => {
  goToDeploymentWithinModel(modelName);
  cy.contains('td', deploymentName)
    .closest('tr')
    .find('td:last-child')
    .find('button')
    .eq(0)
    .click()
    .wait(300);

  fillForm(deploymentFormData);

  cy.intercept({
    method: 'PUT',
    url: `${API_BASE_URL}/api/v1/deployments/*`,
  }).as('updateDeployment');
  cy.contains('button', 'SAVE').click().wait(300);
  deploymentFormData.shareStrategy === 'Public' &&
    cy.contains('button', 'Confirm').click().wait(300);
  return cy.wait('@updateDeployment').then((interception) => {
    assert.equal(interception.response?.statusCode, 200);
    return cy.wrap(interception.response);
  });
};

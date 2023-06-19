import { runAction } from './create-update';
import { goToDeploymentWithinModel } from './find-deployment';

const API_BASE_URL = Cypress.env('API_BASE_URL');

export const deleteDeployment = (modelName: string, deploymentName: string) => {
  goToDeploymentWithinModel(modelName);
  runAction(deploymentName, 2);

  cy.intercept({
    method: 'DELETE',
    url: `${API_BASE_URL}/api/v1/deployments/*`,
  }).as('deleteDeployment');
  cy.contains('button', 'Confirm').click().wait(300);
  return cy.wait('@deleteDeployment').then((interception) => {
    assert.equal(interception.response?.statusCode, 200);
    return cy.wrap(interception.response);
  });
};

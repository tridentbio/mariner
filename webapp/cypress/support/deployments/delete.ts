const API_BASE_URL = Cypress.env('API_BASE_URL');

export const deleteDeployment = (modelName: string, deploymentName: string) => {
  cy.goToDeploymentWithinModel(modelName);
  cy.runAction(deploymentName, 2);

  cy.intercept({
    method: 'DELETE',
    url: `${API_BASE_URL}/api/v1/deployments/*`,
  }).as('deleteDeployment');
  cy.contains('button', 'Confirm').click().wait(300);
  cy.wait('@deleteDeployment', {responseTimeout: 40000}).then((interception) => {
    assert.equal(interception.response?.statusCode, 200);
  });
};

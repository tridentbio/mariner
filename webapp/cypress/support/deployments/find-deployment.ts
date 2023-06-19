const API_BASE_URL = Cypress.env('API_BASE_URL');

export const goToSpecificDeployment = (
  name: string,
  tab: 'All' | 'Public' | 'Shared' | 'My' = 'All'
) => {
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/deployments/?page=0*`,
  }).as('getDeployments');
  cy.visit('/deployments');
  cy.wait('@getDeployments').wait(300);

  cy.contains('button', tab, { timeout: 10000 })
    .click({ force: true })
    .wait(1000);

  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/deployments/*`,
  }).as('getDeployment');
  cy.contains('a', name).click();
  cy.wait('@getDeployment').wait(500);
};

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

export const goToPublicDeployment = (shareUrl: string) => {
  const token = shareUrl.split('public-model/')[1];

  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/deployments/public/*`,
  }).as('getDeployment');
  cy.visit('/public-model/' + token);
  cy.wait('@getDeployment').wait(500);
};

export const openDeploymentInCurrentTable = (name: string) => {
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/deployments/*`,
  }).as('getDeployment');
  cy.contains('a', name).click();
  cy.wait('@getDeployment').wait(500);
};

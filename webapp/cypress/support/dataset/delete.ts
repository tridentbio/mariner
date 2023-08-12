const API_BASE_URL = Cypress.env('API_BASE_URL');

export const deleteDatasetIfAlreadyExists = (datasetName: string) => {
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/datasets/*`,
  }).as('getDatasets');
  cy.intercept({
    method: 'DELETE',
    url: `${API_BASE_URL}/api/v1/datasets/*`,
  }).as('delete');
  cy.visit('/datasets');
  cy.wait('@getDatasets');

  cy.get(`a`).each(($link) => {
    if ($link.text() === datasetName) {
      cy.get('a').contains(datasetName).click();
      cy.get('button[id="delete-dataset"]').click();
      cy.wait('@delete');
    }
  });
};

export const deleteAllDatasets = () => {
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/datasets/?page=0&perPage=25`,
  }).as('getDatasets');
  cy.wait(['@getDatasets']);
  cy.get('tbody a').each((node) => deleteDatasetIfAlreadyExists(node.text()));
};

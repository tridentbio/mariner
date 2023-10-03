import { createRandomDatasetFormData } from '../../support/dataset/examples';

const API_BASE_URL = Cypress.env('API_BASE_URL');

describe('/datasets/new - Dataset creation page', () => {
  before(() => {
    cy.loginUser('admin', 15000);
  })

  beforeEach(() => {
    cy.visit('/datasets');

    cy.intercept({
      method: 'GET',
      url: `${API_BASE_URL}/api/v1/datasets/?*`,
    }).as('getDatasets');
    cy.wait('@getDatasets').then(({ response }) => {
      expect(response?.statusCode).to.eq(200);
    });
    cy.get('button', { timeout: 2000 }).contains('Add Dataset').click();
  });

  it('Creates datasets sucessfully', () => {
    const datasetForm = createRandomDatasetFormData();
    cy.createDataset(datasetForm);
  });
});

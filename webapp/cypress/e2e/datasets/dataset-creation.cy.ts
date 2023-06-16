import { createRandomDatasetFormData } from '../../support/dataset/examples';

describe('/datasets/new - Dataset creation page', () => {
  beforeEach(() => {
    cy.loginSuper(15000);
    cy.visit('/datasets');

    cy.intercept({
      method: 'GET',
      url: '/api/v1/datasets/?*',
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

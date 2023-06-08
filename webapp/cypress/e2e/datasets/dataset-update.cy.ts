import { randomLowerCase } from 'utils';
import { DATASET_NAME } from '../../support/constants';
import { DatasetFormData } from '../../support/dataset/create';
import { deleteDatasetIfAlreadyExists } from '../../support/dataset/delete';

describe('/datasets/:datasetId/edit - Dataset Edit Page', () => {
  const updatedDataset = {
    name: randomLowerCase(),
    description: randomLowerCase(24),
  };
  let dataset: DatasetFormData;
  before(() => {
    cy.loginSuper();

    cy.setupZincDataset().then((zinc) => {
      dataset = zinc;
    });
  });
  after(() => {
    deleteDatasetIfAlreadyExists(updatedDataset.name);
  });
  beforeEach(() => {
    cy.intercept('GET', 'http://localhost:3000/api/v1/datasets/*').as(
      'getDatasets'
    );
    cy.loginSuper();
    cy.visit('/datasets');

    cy.get('tbody', { timeout: 10000 })
      .within((_scope) => {
        cy.get('a').contains(dataset.name).click();
      })
      .wait(300);
    cy.url().should('match', /.*\/datasets\/\d+/);
    cy.get('button').contains('Edit').click().wait(300);
  });
  it('should render for existing dataset', () => {
    cy.url().should('match', /.*\/datasets\/\d+\/edit/);
  });

  it('should update dataset correctly', () => {
    cy.get('#dataset-name-input').clear().type(updatedDataset.name);
    cy.get('#description-input textarea')
      .clear()
      .type(updatedDataset.description);
    cy.get('button').contains('Save').click();
    cy.url().should('match', /^.*\/datasets\/\d+$/);
    cy.get('#dataset-name').should('have.text', updatedDataset.name);
    cy.get('#dataset-description').should(
      'have.text',
      updatedDataset.description
    );
  });
});

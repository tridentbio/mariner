import createDataset from '../../support/dataset/create';
import { deleteDatasetIfAlreadyExists } from '../../support/dataset/delete';
import {
  createRandomDatasetFormData,
  zincDatasetFixture,
} from '../../support/dataset/examples';
import { deleteTestModelsIfExist } from '../../support/models/common';

describe('/models/new - Model creation page', () => {
  const datasetFixture = createRandomDatasetFormData();
  before(() => {
    cy.loginSuper();
  });

  beforeEach(() => {
    cy.visit('/models/new');
  });

  it('Builds Smiles-Numeric regressor', () => {
    cy.buildNumSmilesModel(['smiles', 'mwt'], 'tpsa').then(() => {
      // cy.get('button').contains('CREATE MODEL').click({ force: true });
      // cy.url({ timeout: 60000 }).should('include', `#newtraining`, {
      //   timeout: 60000,
      // });
    });
  });

  it.skip('Builds Smiles-Categorical regressor', () => {
    cy.fixture('models/schemas/');
    cy.buildCategoricalSmilesModel(['smiles', 'mwt_group'], 'tpsa').then(() => {
      cy.get('button').contains('CREATE MODEL').click({ force: true });
      cy.url({ timeout: 60000 }).should('include', `#newtraining`, {
        timeout: 60000,
      });
    });
  });
});

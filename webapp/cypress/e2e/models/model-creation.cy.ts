import createDataset, {
  createDatasetDirectly,
} from '../../support/dataset/create';
import { deleteDatasetIfAlreadyExists } from '../../support/dataset/delete';
import {
  createRandomDatasetFormData,
  zincDatasetFixture,
} from '../../support/dataset/examples';
import { deleteTestModelsIfExist } from '../../support/models/common';

describe('/models/new - Model creation page', () => {
  const datasetFixture = createRandomDatasetFormData();
  before(async () => {
    cy.loginSuper();
    await createDatasetDirectly(datasetFixture);
  });

  beforeEach(() => {
    cy.visit('/models/new');
  });

  it('Builds Smiles-Numeric regressor', () => {
    cy.buildNumSmilesModel(datasetFixture.name).then(() => {});
  });

  it.skip('Builds Smiles-Categorical regressor', () => {
    cy.fixture('models/schemas/');
    cy.buildCategoricalSmilesModel(
      ['smiles', 'mwt_group'],
      'tpsa',
      datasetFixture.name
    ).then(() => {
      cy.get('button').contains('CREATE MODEL').click({ force: true });
      cy.url({ timeout: 60000 }).should('include', `#newtraining`, {
        timeout: 60000,
      });
    });
  });
});

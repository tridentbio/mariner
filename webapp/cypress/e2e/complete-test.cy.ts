import {
  MODEL_SMILES_CATEGORICAL_NAME,
  MODEL_SMILES_NUMERIC_NAME,
} from '../support/constants';
import createDataset from '../support/dataset/create';
import { deleteDatasetIfAlreadyExists } from '../support/dataset/delete';
import { zincDatasetFixture } from '../support/dataset/examples';
import { deleteTestModelsIfExist } from '../support/models/common';
import { checkModelTraining } from '../support/training/create';

describe.skip('Complete test from dataset creation to inference', () => {
  before(() => {
    cy.loginSuper();
    deleteDatasetIfAlreadyExists(zincDatasetFixture.name);
    deleteTestModelsIfExist();
  });
  after(() => {
    cy.loginSuper();
    deleteDatasetIfAlreadyExists(zincDatasetFixture.name);
    deleteTestModelsIfExist();
  });

  it('Should create dataset, create models, train models and run inference succesfully', () => {
    createDataset(zincDatasetFixture);
    cy.buildYamlModel(
      'data/yaml/categorical_features_model.yaml',
      zincDatasetFixture.name
    );
    cy.buildYamlModel(
      'models/schemas/small_regressor_schema.yaml',
      zincDatasetFixture.name
    );
    checkModelTraining(MODEL_SMILES_CATEGORICAL_NAME);
    checkModelTraining(MODEL_SMILES_NUMERIC_NAME);
  });
});
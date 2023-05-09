import { randomLowerCase } from '@utils';
import { MODEL_SMILES_CATEGORICAL_NAME } from '../../support/constants';
import { createDatasetDirectly } from '../../support/dataset/create';
import { createIrisDatasetFormData } from '../../support/dataset/examples';
import { checkModelTraining } from '../../support/training/create';

// set following variables to skip model build and
// use this models 1st versions as fixtures

describe('Model Training Page', () => {
  const irisDatasetFixture = createIrisDatasetFormData();
  const modelName = randomLowerCase(8);

  beforeEach(() => {
    cy.loginSuper();
    cy.then(() => createDatasetDirectly(irisDatasetFixture));
    cy.buildYamlModel(
      'data/yaml/binary_classification_model.yaml',
      irisDatasetFixture.name,
      true,
      true,
      modelName
    );
  });
  it('Creates training succesfully', () => {
    checkModelTraining(modelName);
    // TODO: Asserts the progress bar fills
    // TODO: Asserts loss goes down
  });
  // TODO: Implement specific models training
  // it('Creates training succesfully for a model with categorical and smiles inputs', () => {
  //   checkModelTraining(modelSmilesCategoricalFixture);
  // });

  // it('Creates training successfully for a model with numerical and smiles inputs', () => {
  //   checkModelTraining(modelSmilesNumFixture);
  // });
});

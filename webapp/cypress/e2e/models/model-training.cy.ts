import { MODEL_SMILES_CATEGORICAL_NAME } from '../../support/constants';
import { checkModelTraining } from '../../support/training/create';

// set following variables to skip model build and
// use this models 1st versions as fixtures

describe('Model Training Page', () => {
  beforeEach(() => {
    cy.loginSuper();
  });
  it('Creates training succesfully', () => {
    checkModelTraining();
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

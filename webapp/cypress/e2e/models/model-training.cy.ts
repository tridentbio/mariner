import { createIrisDatasetFormData } from '../../support/dataset/examples';
import { trainModel } from '../../support/training/create';

// set following variables to skip model build and
// use this models 1st versions as fixtures

describe('Model Training Page', () => {
  const irisDatasetFixture = createIrisDatasetFormData();
  let modelName: string | null = null;

  beforeEach(() => {
    cy.loginSuper();
    cy.setupSomeModel().then((name) => {
      modelName = name;
    });
  });
  it('Creates training succesfully', () => {
    trainModel(modelName!).then((experimentName) => {
      assert.isNotNull(experimentName);
    });

    // TODO: Asserts the progress bar fills
    // TODO: Asserts loss goes down
  });
  // TODO: Implement specific models training
  // it('Creates training succesfully for a model with categorical and smiles inputs', () => {
  //   trainModel(modelSmilesCategoricalFixture);
  // });

  // it('Creates training successfully for a model with numerical and smiles inputs', () => {
  //   trainModel(modelSmilesNumFixture);
  // });
});

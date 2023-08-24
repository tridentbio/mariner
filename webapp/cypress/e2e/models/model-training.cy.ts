import { trainModel } from '../../support/training/create';

// set following variables to skip model build and
// use this models 1st versions as fixtures

describe('Model Training Page', () => {
  let modelName: string | null = null;
  let modelId: number | null = null
  beforeEach(() => {
    cy.loginSuper();
    cy.setupSomeModel().then((deployment) => {
      modelName = deployment.name;
      if ('id' in deployment)
        modelId = deployment.id
    });
  });
  it('Creates training succesfully', () => {
    trainModel(modelName!).then((experiment: any) => {
      assert.isNotNull(experiment);
      // Assert metrics are found in metrics table
      cy.contains('button', 'Metrics', {timeout: 3000}).should('exist').click()
      cy.get('#model-version-select').click().get('li[role="option"]').contains(experiment.modelVersion.name).click()
      cy.get('[data-testid="experiment-select"] div').click().get('li[role="option"]').contains(experiment.experimentName).click()
      cy.get('table').contains('Train')
      cy.get('table').contains('Validation')
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

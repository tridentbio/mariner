import { createDeployment } from '../../support/deployments/create';
import { trainModel } from '../../support/training/create';
// set following variables to skip model build and
// use this models 1st versions as fixtures

describe('Model Training Page', () => {
  let modelName: string | null = null;
  let modelVersionName: string | null = null;
  let deploymentName: string | null = null;

  before(() => {
    cy.loginSuper();
    cy.setupSomeModel().then((deployment) => {
      modelName = deployment.name;
      modelVersionName = deployment.config.name;
    });
    trainModel(modelName!, { modelVersion: modelVersionName! }).then(
      (experimentName) => {
        assert.isNotNull(experimentName);
      }
    );
  });

  it('Creates training succesfully', () => {
    createDeployment(modelName!, modelVersionName!).then((name) => {
      deploymentName = name;
    });
  });

  it('Share test with test user', () => {
    cy.loginTest();
    cy.visit('/deployments');
    cy.contains('a', deploymentName!).click();
    cy.get('button').contains('Share').click().wait(300);
  });
});

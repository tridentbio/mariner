import {
  goToPublicDeployment,
  goToSpecificDeployment,
} from '../../support/deployments/find-deployment';
import {
  createDeployment,
  updateDeployment,
} from '../../support/deployments/create-update';

const TEST_USER = Cypress.env('TEST_USER');

describe('Deployments Page', () => {
  let modelName: string | null = null;
  let modelVersionName: string | null = null;
  let deploymentName: string | null = null;

  before(() => {
    cy.loginSuper();
    cy.setupSomeModel().then((deployment) => {
      modelName = deployment.name;
      modelVersionName = deployment.config.name;
    });
  });

  it('Deploys model succesfully', () => {
    createDeployment(modelName!, modelVersionName!).then((name) => {
      deploymentName = name;
    });
  });

  it('Share model with test user', () => {
    cy.loginSuper();
    updateDeployment(modelName!, deploymentName!, {
      shareWithUser: [TEST_USER],
    });

    cy.loginTest();
    goToSpecificDeployment(deploymentName!, 'Shared');
  });

  it('Access model publically', () => {
    cy.loginSuper();
    updateDeployment(modelName!, deploymentName!, {
      shareStrategy: 'Public',
    }).then((res) => {
      const shareUrl = res?.body.shareUrl as string;
      assert.exists(shareUrl);
      goToPublicDeployment(shareUrl!);
    });
  });
});

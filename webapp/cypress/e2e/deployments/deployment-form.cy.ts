const TEST_USER = Cypress.env('TEST_USER');

describe('Deployments Form.', () => {
  let modelName: string | null = null;
  let modelVersionName: string | null = null;
  let deploymentName: string | null = null;

  before(() => {
    cy.loginSuper();
    cy.setupSomeModel().then((model) => {
      modelName = model.name;
      modelVersionName =  'config' in model ? model.config.name : model.versions[0].config.name
    });
  });

  after(() => {
    cy.loginSuper();
    cy.deleteDeployment(modelName!, deploymentName!);
  });

  it('Deploys model succesfully.', () => {
    cy.createDeployment(modelName!, modelVersionName!).then((name) => {
      deploymentName = name;
    });
  });

  it('Share model with test user.', () => {
    cy.loginSuper();
    cy.updateDeployment(modelName!, deploymentName!, {
      shareWithUser: [TEST_USER],
    });

    cy.loginTest();
    cy.goToSpecificDeployment(deploymentName!, 'Shared');
  });

  it('Access model publically.', () => {
    cy.loginSuper();
    cy.updateDeployment(modelName!, deploymentName!, {
      shareStrategy: 'Public',
    }).then((res) => {
      const shareUrl = res?.body.shareUrl as string;
      assert.exists(shareUrl);
      cy.goToPublicDeployment(shareUrl!);
    });
  });
});

const TEST_USER = Cypress.env('TEST_USER');

describe('Deployments Form.', () => {
  let modelName: string | null = null;
  let modelVersionName: string | null = null;
  let deploymentName: string | null = null;

  before(() => {
    cy.loginUser();
    cy.setupSomeModel().then((model) => {
      modelName = model.name;
      modelVersionName =  'config' in model ? model.config.name : model.versions[0].config.name
    });
  });

  after(() => {
    cy.deleteDeployment(modelName!, deploymentName!);
  });

  it('Deploys model succesfully.', () => {
    cy.createDeployment(modelName!, modelVersionName!).then((name) => {
      deploymentName = name;
    });
  });

  it('Share model with test user.', () => {
    cy.loginUser();
    cy.updateDeployment(modelName!, deploymentName!, {
      shareWithUser: [TEST_USER.username],
    });

    cy.loginUser('test');
    cy.goToSpecificDeployment(deploymentName!, 'Shared');
  });

  it('Access model publically.', () => {
    cy.loginUser();
    cy.updateDeployment(modelName!, deploymentName!, {
      shareStrategy: 'Public',
    }).then((res) => {
      const shareUrl = res?.body.shareUrl as string;
      assert.exists(shareUrl);
      cy.goToPublicDeployment(shareUrl!);
    });
  });
});

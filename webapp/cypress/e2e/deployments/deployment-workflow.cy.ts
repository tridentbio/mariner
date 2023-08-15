import { trainModel } from '../../support/training/create';

describe('Deployments Workflow.', () => {
  let modelName: string | null = null;
  let modelVersionName: string | null = null;
  let deploymentName: string | null = null;

  before(() => {
    cy.on(
      'uncaught:exception',
      (err) => err.toString().includes('ResizeObserver') && false
    );
    
    cy.loginSuper();

    cy.setupSomeModel().then((model) => {
      modelName = model.name;
      modelVersionName = model.config.name;

      trainModel(modelName, { modelVersion: modelVersionName }).then(
        (experimentName) => {
          assert.isNotNull(experimentName);
        }
      );

      cy.createDeployment(modelName!, modelVersionName!).then((name) => {
        deploymentName = name;
      });
    });
  });

  after(() => {
    cy.deleteDeployment(modelName!, deploymentName!);
  });

  beforeEach(() => {
    cy.loginSuper();
  });

  it('Runs deployment succesfully.', () => {
    cy.goToDeploymentWithinModel(modelName!);
    cy.handleStatus(deploymentName!, 'idle');
    cy.startDeployment(deploymentName!);
  });

  it('Stops deployment succesfully.', () => {
    cy.goToDeploymentWithinModel(modelName!);
    cy.handleStatus(deploymentName!, 'active');
    cy.stopDeployment(deploymentName!);
  });

  it('Make prediction on stopped deployment with error.', () => {
    cy.goToDeploymentWithinModel(modelName!);
    cy.handleStatus(deploymentName!, 'idle');
    cy.openDeploymentInCurrentTable(deploymentName!);
    cy.makePrediction(false);
    cy.notificationShouldContain('Deployment instance not running.');
  });

  it('Make prediction on active deployment succesfully.', () => {
    cy.goToDeploymentWithinModel(modelName!);
    cy.handleStatus(deploymentName!, 'active');
    cy.openDeploymentInCurrentTable(deploymentName!);
    cy.makePrediction(true);
  });

  it('Make prediction until reach rate limit.', () => {
    cy.goToDeploymentWithinModel(modelName!);
    cy.handleStatus(deploymentName!, 'active');
    cy.openDeploymentInCurrentTable(deploymentName!);
    for (let i = 0; i < 11; i++) {
      // deployments have a rate limit of 10 requests per minute
      cy.makePrediction(false);
    }
    cy.notificationShouldContain(
      'You have reached the prediction limit for this deployment.'
    );

    it('Make prediction on public deployment.', () => {
      cy.goToDeploymentWithinModel(modelName!);

      cy.updateDeployment(modelName!, deploymentName!, {
        shareStrategy: 'Public',
      }).then((res) => {
        const shareUrl = res?.body.shareUrl as string;
        assert.exists(shareUrl);
        cy.goToPublicDeployment(shareUrl!);
        cy.makePrediction(true);
      });
    });
  });
});

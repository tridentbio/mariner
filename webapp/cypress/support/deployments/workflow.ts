type Status = 'stopped' | 'active' | 'starting' | 'idle';

const getDeploymentStatus = (deploymentName: string) =>
  cy
    .contains('td', deploymentName)
    .closest('tr')
    .find('td')
    .eq(2)
    .then((element) => element.text() as Status);

const waitUntilDeploymentStatus = (
  deploymentName: string,
  waitingStatus: Status,
  timeout = 15000
): Cypress.Chainable<boolean> =>
  // Recursively check if the deployment status is equal to the given status
  getDeploymentStatus(deploymentName).then((status) => {
    if (status === waitingStatus) return cy.wrap(true);
    else if (timeout <= 0) return cy.wrap(false);
    else
      return cy
        .wait(1000)
        .then(() =>
          waitUntilDeploymentStatus(
            deploymentName,
            waitingStatus,
            timeout - 1000
          )
        );
  });

export const startDeployment = (deploymentName: string) => {
  getDeploymentStatus(deploymentName).then((s) => {
    assert.notEqual(s, 'active', 'Deployment is already active');
  });

  cy.runAction(deploymentName, 1);

  waitUntilDeploymentStatus(deploymentName, 'active').then((isSuccess) => {
    assert.isTrue(isSuccess, 'Deployment is not active');
  });
};

export const stopDeployment = (deploymentName: string) => {
  getDeploymentStatus(deploymentName).then((s) => {
    assert.equal(s, 'active', 'Deployment is already stopped');
  });

  cy.runAction(deploymentName, 1);

  waitUntilDeploymentStatus(deploymentName, 'idle').then((isSuccess) => {
    assert.isTrue(isSuccess, 'Deployment is not stopped');
  });
};

export const handleStatus = (
  deploymentName: string,
  status: 'active' | 'idle' // possible status to handle manually
) =>
  getDeploymentStatus(deploymentName).then((currentStatus) => {
    const running = currentStatus === 'active';

    if (running && status === 'idle') {
      cy.runAction(deploymentName, 1);
      waitUntilDeploymentStatus(deploymentName, 'idle').then((isSuccess) => {
        assert.isTrue(isSuccess, 'Deployment is not stopped');
      });
    } else if (!running && status === 'active') {
      cy.runAction(deploymentName, 1);
      waitUntilDeploymentStatus(deploymentName, 'active').then((isSuccess) => {
        assert.isTrue(isSuccess, 'Deployment is not active');
      });
    }
  });

export const makePrediction = (expectSuccess: boolean) => {
  cy.contains('div', 'Input')
    .parent()
    .get('input')
    .each(($el) => {
      cy.wrap($el).type(Math.random().toPrecision(3));
    });

  cy.intercept({
    method: 'POST',
    url: 'http://localhost/api/v1/deployments/*/predict',
  }).as('makePrediction');
  cy.get('button').contains('Predict').click();
  cy.wait('@makePrediction').then(({ response }) => {
    if (expectSuccess) {
      expect(response?.statusCode).to.eq(200);
      cy.get('[data-testid="inference-result"]', { timeout: 10000 }).should(
        'exist'
      );
      return cy.wrap(true);
    }
    return cy.wrap(response?.statusCode === 200);
  });
};

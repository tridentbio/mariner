type Status = 'stopped' | 'active' | 'starting' | 'idle';

const API_BASE_URL = Cypress.env('API_BASE_URL');

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
) =>
  cy
    .contains('td', deploymentName)
    .closest('tr')
    .contains('td', waitingStatus, { timeout });

export const startDeployment = (deploymentName: string) => {
  getDeploymentStatus(deploymentName).then((s) => {
    assert.notEqual(s, 'active', 'Deployment is already active');
  });

  cy.runAction(deploymentName, 1);

  waitUntilDeploymentStatus(deploymentName, 'active');
};

export const stopDeployment = (deploymentName: string) => {
  getDeploymentStatus(deploymentName).then((s) => {
    assert.equal(s, 'active', 'Deployment is already stopped');
  });

  cy.runAction(deploymentName, 1);

  waitUntilDeploymentStatus(deploymentName, 'idle');
};

export const handleStatus = (
  deploymentName: string,
  status: 'active' | 'idle' // possible status to handle manually
) =>
  getDeploymentStatus(deploymentName).then((currentStatus) => {
    const running = currentStatus === 'active';

    if (running && status === 'idle') {
      cy.runAction(deploymentName, 1);
      waitUntilDeploymentStatus(deploymentName, 'idle');
    } else if (!running && status === 'active') {
      cy.runAction(deploymentName, 1);
      waitUntilDeploymentStatus(deploymentName, 'active');
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
    url: `${API_BASE_URL}/api/v1/deployments/*/predict*`,
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

import { addDescription } from '../commands';

export interface DatasetFormData {
  name: string;
  description: string;
  file: string;
  splitType: 'Random' | 'Scaffold';
  splitColumn: string;
  descriptions: {
    pattern: string;
    dataType: {
      domainKind: 'Numeric' | 'SMILES' | 'Categorical' | 'String';
      unit?: string;
    };
    description: string;
  }[];
  split: string;
}
const createDataset = (dataset: DatasetFormData) => {
  cy.once('uncaught:exception', () => false);
  cy.intercept({
    method: 'POST',
    url: 'http://localhost/api/v1/datasets/',
  }).as('createDataset');
  cy.intercept({
    method: 'GET',
    url: 'http://localhost/api/v1/units/?q=',
  }).as('getUnits');
  cy.get('#dataset-name-input').type(dataset.name);
  cy.get('#description-input textarea').type(dataset.description);
  cy.get('#dataset-upload').attachFile('zinc.csv');
  cy.wait('@getUnits', { timeout: 2000000 }).then(({ response }) => {
    expect(response?.statusCode).to.eq(200);
  });
  cy.get('#dataset-splittype-input')
    .click()
    .get('li')
    .contains(dataset.splitType)
    .click();
  if (dataset.splitType !== 'Random') {
    cy.get('#dataset-split-column-input')
      .click()
      .get('li')
      .contains(dataset.splitColumn)
      .click();
  }
  cy.get('#dataset-split-input').type(dataset.split);

  dataset.descriptions.forEach(({ pattern, dataType, description }) => {
    addDescription(pattern, dataType, description);
  });

  cy.get('button[id="save"]').click();

  cy.wait('@createDataset').then(({ response }) => {
    expect(response?.statusCode).to.eq(200);
    expect(response?.body).to.have.property('id');
    expect(response?.body.name).to.eq(dataset.name);
  });

  cy.url({ timeout: 30000 }).should('include', `/datasets`, {
    timeout: 30000,
  });
};

export default createDataset;

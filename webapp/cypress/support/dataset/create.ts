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
      classes?: any;
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

  dataset.descriptions?.forEach(({ pattern, dataType, description }) => {
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

const everyLowerCase = <T extends Record<string, any>>(obj: T): T =>
  Object.keys(obj).reduce((acc, key: keyof T) => {
    const value = obj[key];
    if (typeof value === 'string') {
      acc[key] = value.toLowerCase();
    } else if (typeof value === 'object') {
      acc[key] = everyLowerCase(value);
    }
    return acc;
  }, {} as T);

export const createDatasetDirectly = (dataset: DatasetFormData) => {
  cy.once('uncaught:exception', () => false);

  cy.wrap(
    new Promise<string>((res, rej) =>
      cy.window().then((win) => {
        const value = JSON.parse(win.localStorage.getItem('app-token') || '{}');
        if (!value.access_token) {
          rej('No token found yet');
        }
        res(`Bearer ${value.access_token}`);
      })
    )
  ).then((token) => {
    cy.wrap(
      new Promise<Blob>((res) => cy.fixture(dataset.file, 'binary').then(res))
    ).then((file) => {
      const boundary = `----${Math.random().toString().slice(2, 16)}`;

      // FormData is not supported by Cypress and needs to be created manually
      const formData =
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="name"\r\n\r\n` +
        `${dataset.name}\r\n` +
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="columnsMetadata"\r\n\r\n` +
        `${JSON.stringify(dataset.descriptions.map(everyLowerCase))}\r\n` +
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="file"; filename="${dataset.file}"\r\n` +
        `Content-Type: text/csv\r\n\r\n` +
        `${file}\r\n` +
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="splitTarget"\r\n\r\n` +
        `${dataset.split}\r\n` +
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="splitType"\r\n\r\n` +
        `${dataset.splitType.toLowerCase()}\r\n` +
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="splitOn"\r\n\r\n` +
        `${dataset.splitType === 'Random' ? '' : dataset.splitColumn}\r\n` +
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="description"\r\n\r\n` +
        `${dataset.description}\r\n` +
        `--${boundary}--\r\n`;

      cy.wrap(
        new Promise<void>((res) =>
          cy
            .request({
              method: 'POST',
              url: 'http://localhost/api/v1/datasets/',
              body: formData,
              headers: {
                'Content-Type': `multipart/form-data; boundary=${boundary}`,
                authorization: token,
              },
            })
            .then((response) => {
              expect(response.status).to.eq(200);
              res();
            })
        )
      ).then(() => {
        cy.wrap(new Promise<void>((res) => cy.wait(10000).then(res)));
      });
    });
  });
};
export default createDataset;

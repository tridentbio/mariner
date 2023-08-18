import { addDescription } from '../commands';
import { irisDatasetFixture, zincDatasetFixture } from './examples';

const API_BASE_URL = Cypress.env('API_BASE_URL');
const SCHEMA_PATH = Cypress.env('SCHEMA_PATH');

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
    url: `${API_BASE_URL}/api/v1/datasets/`,
  }).as('createDataset');
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/units/?q=`,
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

  return cy.url({ timeout: 30000 }).should('include', `/datasets`, {
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

const sliced = (content: string, numRows: number) => {
  const lines = content.split('\n');
  const header = lines[0];
  const body = lines.slice(1, numRows + 1);
  return [header, ...body].join('\n');
};

class FormData {
  private form: string;
  private boundary: string;
  constructor() {
    this.boundary = `----${Math.random().toString().slice(2, 16)}`;
    this.form = '';
  }

  private parseArgs(filename?: string, contentType?: string): string {
    let args = '';
    if (filename) {
      args += `; filename="${filename}"`;
    }
    if (contentType) {
      args += `\r\nContent-Type: ${contentType}`;
    }
    args += '\r\n\r\n';
    return args;
  }

  public setFormValue(
    key: string,
    value: string | any,
    args: {
      filename?: string;
      contentType?: string;
    } = {}
  ): void {
    this.form +=
      `--${this.boundary}\r\n` +
      `Content-Disposition: form-data; name="${key}"` +
      this.parseArgs(args.filename, args.contentType) +
      `${value}\r\n`;
  }

  public getBody(): string {
    return this.form + `--${this.boundary}--\r\n`;
  }

  public getHeaders(authorization: string): Record<string, string> {
    return {
      authorization,
      'Content-Type': `multipart/form-data; boundary=${this.boundary}`,
    };
  }
}

export const createDatasetDirectly = (
  dataset: DatasetFormData,
  numRows: number = 10
) => {
  cy.once('uncaught:exception', () => false);

  cy.getCurrentAuthString().then((authorization) => {
    cy.readFile<string>(SCHEMA_PATH + dataset.file)
    .then((file) => {
      const formData = new FormData();
      formData.setFormValue('name', dataset.name);
      formData.setFormValue(
        'columnsMetadata',
        JSON.stringify(dataset.descriptions.map(everyLowerCase))
      );
      formData.setFormValue('file', sliced(file, numRows), {
        filename: dataset.file,
        contentType: 'text/csv',
      });
      formData.setFormValue('splitTarget', dataset.split);
      formData.setFormValue('splitType', dataset.splitType.toLowerCase());
      dataset.splitType !== 'Random' &&
        formData.setFormValue('splitOn', dataset.splitColumn);
      formData.setFormValue('description', dataset.description);

      return cy
        .request({
          method: 'POST',
          url: `${API_BASE_URL}/api/v1/datasets/`,
          body: formData.getBody(),
          headers: formData.getHeaders(authorization as string),
        })
        .then((response) => {
          expect(response.status).to.eq(200);
        });
    });
  });
};

export const datasetExists = (
  dataset: DatasetFormData
): Cypress.Chainable<boolean> =>
  cy.getCurrentAuthString().then((authorization) =>
    cy
      .request({
        method: 'GET',
        url:
          `${API_BASE_URL}/api/v1/datasets?page=0&perPage=100&search_by_name=` +
          dataset.name,
        headers: {
          authorization,
        },
      })
      .then((response) => {
        expect(response?.status).to.eq(200);
        const datasets: any[] = response?.body.data || [];
        return cy.wrap(datasets.some((d) => d.name === dataset.name));
      })
  );

export const setupIrisDatset = () =>
  datasetExists(irisDatasetFixture).then((exists) => {
    cy.on('uncaught:exception', () => false);

    if (!exists) {
      createDatasetDirectly(irisDatasetFixture);
    }
    return cy.wrap(irisDatasetFixture);
  });

export const setupZincDataset = () =>
  datasetExists(zincDatasetFixture).then((exists) => {
    cy.on('uncaught:exception', () => false);

    if (!exists) {
      createDatasetDirectly(zincDatasetFixture);
    }
    return cy.wrap(zincDatasetFixture);
  });

export default createDataset;

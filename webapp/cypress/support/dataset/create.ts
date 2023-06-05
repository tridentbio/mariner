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

  public getHeaders(token: string): Record<string, string> {
    return {
      authorization: token,
      'Content-Type': `multipart/form-data; boundary=${this.boundary}`,
    };
  }
}

export const createDatasetDirectly = (
  dataset: DatasetFormData,
  numRows: number = 10
) => {
  // Creates a dataset directly with cypress promises
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
      const formData = new FormData();
      formData.setFormValue('name', dataset.name);
      formData.setFormValue(
        'columnsMetadata',
        JSON.stringify(dataset.descriptions.map(everyLowerCase))
      );
      formData.setFormValue('file', sliced(file as string, numRows), {
        filename: dataset.file,
        contentType: 'text/csv',
      });
      formData.setFormValue('splitTarget', dataset.split);
      formData.setFormValue('splitType', dataset.splitType.toLowerCase());
      dataset.splitType !== 'Random' &&
        formData.setFormValue('splitOn', dataset.splitColumn);
      formData.setFormValue('description', dataset.description);

      cy.wrap(
        new Promise<void>((res) =>
          cy
            .request({
              method: 'POST',
              url: 'http://localhost/api/v1/datasets/',
              body: formData.getBody(),
              headers: formData.getHeaders(token as string),
            })
            .then((response) => {
              expect(response.status).to.eq(200);
              res();
            })
        )
      );
    });
  });
};
export default createDataset;

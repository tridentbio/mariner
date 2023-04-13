import { randomLowerCase } from 'utils';
import { DATASET_NAME } from '../../support/constants';
import createDataset, { DatasetFormData } from '../../support/dataset/create';
import { deleteDatasetIfAlreadyExists } from '../../support/dataset/delete';

describe('/datasets/:datasetId/edit - Dataset Edit Page', () => {
  const dataset: DatasetFormData = {
    name: DATASET_NAME,
    description: randomLowerCase(24),
    file: 'zinc.csv',
    splitType: 'Random',
    splitColumn: 'smiles',
    split: '60-20-20',
    descriptions: [
      {
        pattern: 'mwt',
        dataType: {
          domainKind: 'Numeric',
          unit: 'mole',
        },
        description: 'A numerical column',
      },
      {
        pattern: 'smiles',
        dataType: {
          domainKind: 'SMILES',
        },
        description: 'A smile column',
      },
      {
        pattern: 'mwt_group',
        dataType: {
          domainKind: 'Categorical',
        },
        description: 'A categorical column',
      },
      {
        pattern: 'tpsa',
        dataType: {
          domainKind: 'Numeric',
          unit: 'mole',
        },
        description: 'Another numerical column',
      },
      {
        pattern: 'zinc_id',
        dataType: {
          domainKind: 'String',
        },
        description: '--',
      },
    ],
  };
  const updatedDataset = {
    name: randomLowerCase(),
    description: randomLowerCase(24),
  };
  before(() => {
    let datasetAlreadyExists = false;
    cy.loginSuper();
    cy.visit('/datasets').wait(2000);
    cy.get(`a`)
      .each(($link) => {
        if ($link.text() === dataset.name) {
          datasetAlreadyExists = true;
        }
      })
      .then(() => {
        if (!datasetAlreadyExists) {
          cy.get('button', { timeout: 10000 }).contains('Add Dataset').click();
          createDataset(dataset);
        }
      });
  });
  after(() => {
    deleteDatasetIfAlreadyExists(updatedDataset.name);
  });
  beforeEach(() => {
    cy.intercept('GET', 'http://localhost:3000/api/v1/datasets/*').as(
      'getDatasets'
    );
    cy.loginSuper();
    cy.visit('/datasets');

    cy.get('tbody', { timeout: 10000 })
      .within((_scope) => {
        cy.get('a').contains(dataset.name).click();
      })
      .wait(300);
    cy.url().should('match', /.*\/datasets\/\d+/);
    cy.get('button').contains('Edit').click().wait(300);
  });
  it('should render for existing dataset', () => {
    cy.url().should('match', /.*\/datasets\/\d+\/edit/);
  });

  it('should update dataset correctly', () => {
    cy.get('#dataset-name-input').clear().type(updatedDataset.name);
    cy.get('#description-input textarea')
      .clear()
      .type(updatedDataset.description);
    cy.get('button').contains('Save').click();
    cy.url().should('match', /^.*\/datasets\/\d+$/);
    cy.get('#dataset-name').should('have.text', updatedDataset.name);
    cy.get('#dataset-description').should(
      'have.text',
      updatedDataset.description
    );
  });
});

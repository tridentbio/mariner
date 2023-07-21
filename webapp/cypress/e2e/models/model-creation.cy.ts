import { DatasetFormData } from '../../support/dataset/create';

const DATA_PATH = Cypress.env('DATA_PATH');

describe('/models/new - Model creation page', () => {
  let zincDatasetFixture: DatasetFormData | null = null;
  let irisDatasetFixture: DatasetFormData | null = null;

  before(() => {
    cy.on(
      'uncaught:exception',
      (err) => err.toString().includes('ResizeObserver') && false
    );

    cy.loginSuper();

    cy.setupIrisDatset().then((iris) => {
      irisDatasetFixture = iris;
    });

    cy.setupZincDataset().then((zinc) => {
      zincDatasetFixture = zinc;
    });
  });

  beforeEach(() => {
    cy.loginSuper();
    cy.visit('/models/new');
  });

  it.skip('Builds Categorical-Smiles Model', () => {
    cy.buildYamlModel(
      'data/yaml/categorical_features_model.yaml',
      zincDatasetFixture!.name,
      true
    );
  });

  it('Builds Binary Classification Model', () => {
    cy.buildYamlModel(
      DATA_PATH + '/yaml/binary_classification_model.yaml',
      irisDatasetFixture!.name,
      true
    );
  });

  it('Builds Multiclass Classification Model', () => {
    cy.buildYamlModel(
      DATA_PATH + '/yaml/multiclass_classification_model.yaml',
      irisDatasetFixture!.name,
      true
    );
  });

  it('Builds Multitarget Model', () => {
    cy.buildYamlModel(
      DATA_PATH + '/yaml/multitarget_classification_model.yaml',
      irisDatasetFixture!.name,
      true
    );
  });
  
});

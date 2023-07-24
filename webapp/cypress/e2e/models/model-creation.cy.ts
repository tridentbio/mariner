import { DatasetFormData } from '../../support/dataset/create';

const SCHEMA_PATH = Cypress.env('SCHEMA_PATH');

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

  // TODO: fix OneHot Layer bug to this test pass
  // it.skip('Builds Categorical-Smiles Model', () => {
  //   cy.buildYamlModel(
  //     SCHEMA_PATH + '/yaml/categorical_features_model.yaml',
  //     zincDatasetFixture.name,
  //     true
  //   );
  // });

  it('Builds Binary Classification Model', () => {
    cy.buildYamlModel(
      SCHEMA_PATH + '/yaml/binary_classification_model.yaml',
      irisDatasetFixture!.name,
      true
    );
  });

  it('Builds Multiclass Classification Model', () => {
    cy.buildYamlModel(
      SCHEMA_PATH + '/yaml/multiclass_classification_model.yaml',
      irisDatasetFixture!.name,
      true
    );
  });

  it('Builds Multitarget Model', () => {
    cy.buildYamlModel(
      SCHEMA_PATH + '/yaml/multitarget_classification_model.yaml',
      irisDatasetFixture!.name,
      true
    );
  });

  it('Builds Smiles-Numeric regressor', () => {
    cy.buildYamlModel(
      SCHEMA_PATH + '/yaml/small_regressor_schema.yaml',
      zincDatasetFixture!.name,
      false
    );
  });
});

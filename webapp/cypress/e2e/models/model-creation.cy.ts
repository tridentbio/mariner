import createDataset, {
  createDatasetDirectly,
} from '../../support/dataset/create';
import { deleteDatasetIfAlreadyExists } from '../../support/dataset/delete';
import {
  createIrisDatasetFormData,
  createRandomDatasetFormData,
} from '../../support/dataset/examples';

describe('/models/new - Model creation page', () => {
  const zincDatasetFixture = createRandomDatasetFormData();
  const irisDatasetFixture = createIrisDatasetFormData();

  before(async () => {
    cy.loginSuper();
    await createDatasetDirectly(zincDatasetFixture);
    await createDatasetDirectly(irisDatasetFixture);
  });

  after(async () => {
    deleteDatasetIfAlreadyExists(zincDatasetFixture.name);
    deleteDatasetIfAlreadyExists(irisDatasetFixture.name);
  });

  beforeEach(() => {
    cy.loginSuper();
    cy.visit('/models/new');
  });

  // TODO: fix OneHot Layer bug to this test pass
  // notion card: https://www.notion.so/tridentbio/BUG-in-One-Hot-Layer-while-creating-a-new-model-44af40309ff94ae28ae439fe19039642
  // it('Builds Categorical-Smiles Model', () => {
  //   cy.buildYamlModel(
  //     'data/yaml/categorical_features_model.yaml',
  //     zincDatasetFixture.name,
  //     true
  //   );
  // });

  it('Builds Binary Classification Model', () => {
    cy.buildYamlModel(
      'data/yaml/binary_classification_model.yaml',
      irisDatasetFixture.name,
      true
    );
  });

  it('Builds Multiclass Classification Model', () => {
    cy.buildYamlModel(
      'data/yaml/multiclass_classification_model.yaml',
      irisDatasetFixture.name,
      true
    );
  });

  it('Builds Multitarget Model', () => {
    cy.buildYamlModel(
      'data/yaml/multitarget_classification_model.yaml',
      irisDatasetFixture.name,
      true
    );
  });

  it('Builds Smiles-Numeric regressor', () => {
    cy.buildYamlModel(
      'models/schemas/small_regressor_schema.yaml',
      zincDatasetFixture.name,
      false
    );
  });
});

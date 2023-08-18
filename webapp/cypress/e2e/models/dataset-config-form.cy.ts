import * as modelsApi from 'app/rtk/generated/models';
import { DeepPartial } from 'react-hook-form';
import { DatasetFormData } from '../../support/dataset/create';
import { fillDatasetCols, fillModelDescriptionStepForm } from '../../support/models/build-model';
import { getColumnConfigTestId } from '@components/organisms/ModelBuilder/utils';

describe('DatasetConfigForm', () => {
  let irisDatasetFixture: DatasetFormData | null = null;

  const testModel: DeepPartial<modelsApi.ModelCreate> = {
    name: 'test_model',
    modelDescription: 'test model description',
    modelVersionDescription: 'test model version description',
    config: {
      name: 'Version name test',
      dataset: {
        targetColumns: [
          {
            name: 'sepal_length',
            dataType: {
              domainKind: 'numeric',
              unit: 'cm',
            },
          },
          {
            name: 'sepal_width',
            dataType: {
              domainKind: 'numeric',
              unit: 'cm',
            },
          },
        ],
        featureColumns: [
          {
            name: 'sepal_length',
            dataType: {
              domainKind: 'numeric',
              unit: 'cm',
            },
          },
          {
            name: 'sepal_width',
            dataType: {
              domainKind: 'numeric',
              unit: 'cm',
            },
          },
        ],
      },
      spec: {
        layers: [],
      },
    },
  };

  const targetCols = testModel.config?.dataset?.targetColumns;
  const featureCols = testModel.config?.dataset?.featureColumns;

  if (!targetCols) throw new Error('targetCols is undefined');
  if (!featureCols) throw new Error('featureCols is undefined');

  before(() => {
    cy.loginSuper();

    cy.setupIrisDatset().then((iris) => {
      irisDatasetFixture = iris;
    });
  });

  beforeEach(() => {
    cy.loginSuper();
    cy.visit('/models/new');

    if (testModel.config?.dataset?.name)
      testModel.config.dataset.name = irisDatasetFixture?.name;
    else (testModel.config as modelsApi.TorchModelSpec).dataset.name = 'Iris';

    fillModelDescriptionStepForm(testModel);

    cy.get('button').contains('NEXT').click();

    cy.get('#dataset-select')
      .click()
      .type(testModel.config?.dataset?.name || '')
      .get('li[role="option"]')
      .first()
      .click();
  });

  it('should not include target columns on the feature columns list', () => {
    fillDatasetCols(targetCols as modelsApi.ColumnConfig[], '#target-col');

    cy.get('#feature-cols').click();

    targetCols.forEach((col) => {
      cy.get('div[role="presentation"]').should('not.contain.text', col?.name);
    });
  });

  it('Should remove feature column options when they are selected as target columns', () => {
    fillDatasetCols(featureCols as modelsApi.ColumnConfig[], '#feature-cols');

    const firstTargetCol = targetCols[0]
    const firstTargetColTestId = getColumnConfigTestId(targetCols[0] as modelsApi.ColumnConfig);

    cy.get('#target-col').click();
    cy.get(`li[data-testid="${firstTargetColTestId}"`).click();

    cy.get('[data-testid="dataset-feature-columns"]').should(
      'not.contain.text',
      firstTargetCol?.name
    );
  });
});

import { ModelBuilderContextProvider } from '@components/organisms/ModelBuilder/hooks/useModelBuilder';
import { getColumnConfigTestId } from '@components/organisms/ModelBuilder/utils';
import ModelCreateV2, { schema } from '@features/models/pages/ModelCreateV2';
import { yupResolver } from '@hookform/resolvers/yup';
import * as modelsApi from 'app/rtk/generated/models';
import { FormProvider, useForm } from 'react-hook-form';
import { DefaultProviders } from '../support/DefaultProviders';
import TestUtils from '../support/TestUtils';
import { fillDatasetCols } from '../support/models/build-model';

describe('ModelCreateV2.cy.tsx', () => {
  const testModel: modelsApi.ModelCreate = {
    name: 'test_model',
    modelDescription: 'test model description',
    modelVersionDescription: 'test model version description',
    config: {
      framework: 'torch',
      name: 'Model name test',
      dataset: {
        name: 'IRIS_DATASET_NAME',
        strategy: 'forwardArgs',
        featureColumns: [
          {
            name: 'sepal_length',
            dataType: {
              domainKind: 'numeric',
              unit: 'mole',
            },
          },
          {
            name: 'sepal_width',
            dataType: {
              domainKind: 'numeric',
              unit: 'mole',
            },
          },
        ],
        targetColumns: [
          {
            name: 'large_petal_length',
            dataType: {
              domainKind: 'categorical',
              classes: {
                '0': 0,
                '1': 1,
              },
            },
            outModule: '',
          },
        ],
      },
      spec: {
        layers: [],
      },
    },
  };

  const MountedComponent = () => {
    const methods = useForm<modelsApi.ModelCreate>({
      mode: 'all',
      reValidateMode: 'onBlur',
      defaultValues: testModel,
      resolver: yupResolver(schema),
    });

    return (
      <DefaultProviders>
        <FormProvider {...methods}>
          <ModelBuilderContextProvider>
            <ModelCreateV2 />
          </ModelBuilderContextProvider>
        </FormProvider>
      </DefaultProviders>
    );
  }

  const targetCols = testModel.config.dataset?.targetColumns;
  const featureCols = testModel.config.dataset?.featureColumns;

  if (!targetCols) throw new Error('targetCols is undefined');
  if (!featureCols) throw new Error('featureCols is undefined');

  const clickNext = () => cy.get(`[data-testid="next"]`).click();

  const clickPrevious = () =>
    cy.get(`[data-testid="previous"]`).click({ timeout: 10000 });

  const goToFirstStep = () =>
    clickPrevious().then(() => {
      cy.get('body').then(($body) => {
        const btn = $body.find('[data-testid="previous"]')
        
        if(btn.length) goToFirstStep();
      })
    })

  const fillModelForm = (
    modelFormData?: typeof testModel,
  ) => {
    if (modelFormData) {
      cy.get('@modelInput')
        .type(modelFormData.name)
        .wait(500)
        .type('{enter}');

      cy.get('[data-testid="model-description"] input').type(
        modelFormData.modelDescription as string
      );

      cy.get('[data-testid="version-name"] input').type(
        modelFormData.config.name
      );

      cy.get('[data-testid="version-description"] textarea').type(
        modelFormData.modelVersionDescription as string
      );

      clickNext();
    }
  };

  const fillDatasetForm = (dataset: modelsApi.ModelCreate['config']['dataset']) => {
    cy.get('#dataset-select')
    .click()
    .type(dataset.name || '')
    .get('li[role="option"]')
    .first()
    .click();

    fillDatasetCols(dataset.targetColumns as modelsApi.ColumnConfig[], '#target-col');
    fillDatasetCols(dataset.featureColumns as modelsApi.ColumnConfig[], '#feature-cols');
  }

  before(() => {
    cy.on(
      'uncaught:exception',
      (err) => err.toString().includes('ResizeObserver') && false
    );
  });

  beforeEach(() => {
    cy.mount(<MountedComponent />);

    cy.get('[data-testid="model-name"] input').as('modelInput');
  })


  it('Model form starts with empty model name and model description', () => {
    cy.get('@modelInput').should('have.value', '');
  });

    it('Model name random generation works', () => {
      cy.get('@modelInput').invoke(
        'text',
        (previousText: string) => {
          cy.get('[data-testid="random-model-name"]').click();

          cy.get('@modelInput')
            .should(
              'not.have.value',
              previousText
            ).and('not.have.value', '0')
        }
      )
    });

    it('Validation on required fields (model name and model version name)', () => {
      cy.get('@modelInput').clear({ force: true });
      cy.get('[data-testid="version-name"] input').clear({ force: true });
      clickNext()
      cy.notificationShouldContain('Missing');
      cy.get('[data-testid="model-name"] label').should(
        'have.class',
        TestUtils.errorClass
      );
      cy.get('[data-testid="version-name"] label').should(
        'have.class',
        TestUtils.errorClass
      );
    });

    it('Is persisted across step transitions', () => {
      fillModelForm(testModel)

      goToFirstStep().then(() => {
        cy.get('[data-testid="model-description"] input').should(
          'have.value',
          testModel.modelDescription
        );
        cy.get('[data-testid="version-name"] input').should(
          'have.value',
          testModel.config.name
        );
        cy.get('[data-testid="version-description"] textarea').should(
          'have.value',
          testModel.modelVersionDescription
        );
        cy.get('@modelInput').should(
          'have.value',
          testModel.name
        );
      });
    });

  it('Is persisted across step transitions pt.2', () => {
    fillModelForm(testModel);
    fillDatasetForm(testModel.config.dataset)

    clickNext()

    clickPrevious().then(() => {
      cy.get('[data-testid="dataset-target-column"]').contains(
        testModel.config.dataset.targetColumns[0].name,
      );
      cy.wrap(
        testModel.config.dataset.featureColumns.forEach((col) => {
          cy.get('[title="Feature Column"] span')
            .contains(col.name)
            .should('exist');
        })
      );

      cy.get('[data-testid="dataset-selector"] input').should(
        'have.value',
        testModel.config.dataset.name
      );
    });
  });

  it('Validate required fields (dataset.name, dataset.targetColumns, dataset.featureColumns)', () => {
    fillModelForm(testModel);
    
    cy.get('#dataset-select').focus().blur();

    cy.get('[data-testid="dataset-selector"] label').should(
      'have.class',
      TestUtils.errorClass
    );

    clickNext();

    cy.notificationShouldContain('Missing dataset name');

    cy.get('#dataset-select')
      .click()
      .type(testModel.config.dataset.name || '')
      .get('li[role="option"]')
      .first()
      .click();

    cy.get('#target-col').focus().blur();
    cy.get('[data-testid="dataset-target-column"] label').should(
      'have.class',
      TestUtils.errorClass
    );
    cy.get('#feature-cols').focus().blur();
    cy.get('[data-testid="dataset-feature-columns"] label').should(
      'have.class',
      TestUtils.errorClass
    );

    clickNext();
    cy.notificationShouldContain('Missing dataset target column selection');
  });

  it('should not include target columns on the feature columns list', () => {
    fillModelForm(testModel);

    cy.get('#dataset-select')
      .click()
      .type(testModel.config.dataset?.name || '')
      .get('li[role="option"]')
      .first()
      .click();

    fillDatasetCols(targetCols as modelsApi.ColumnConfig[], '#target-col');

    cy.get('#feature-cols').click();

    targetCols.forEach((col) => {
      cy.get('div[role="presentation"]').should('not.contain.text', col?.name);
    });
  });

  it('Should remove feature column options when they are selected as target columns', () => {
    fillModelForm(testModel);

    cy.get('#dataset-select')
      .click()
      .type(testModel.config.dataset?.name || '')
      .get('li[role="option"]')
      .first()
      .click();

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

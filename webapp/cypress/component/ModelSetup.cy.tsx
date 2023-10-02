import { store } from '@app/store';
import { ModelBuilderContextProvider } from '@components/organisms/ModelBuilder/hooks/useModelBuilder';
import { getColumnConfigTestId } from '@components/organisms/ModelBuilder/utils';
import { schema } from '@features/models/pages/ModelCreateV2';
import { ModelSetup } from '@features/models/pages/ModelCreateV2/ModelSetup';
import { yupResolver } from '@hookform/resolvers/yup';
import { ThemeProvider } from '@mui/material';
import * as modelsApi from 'app/rtk/generated/models';
import { DeepPartial, FormProvider, useForm } from 'react-hook-form';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { theme } from 'theme';
import { fillDatasetCols } from '../support/models/build-model';

describe('ModelSetup.cy.tsx', () => {
  const testModel: DeepPartial<modelsApi.ModelCreate> = {
    name: 'test_model',
    modelDescription: 'test model description',
    modelVersionDescription: 'test model version description',
    config: {
      name: 'Version name test',
      dataset: {
        name: 'IRIS_DATASET_NAME',
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

  const MountedComponent = () => {
    const methods = useForm<modelsApi.ModelCreate>({
      mode: 'all',
      reValidateMode: 'onBlur',
      defaultValues: testModel,
      resolver: yupResolver(schema),
    });

    return (
      <BrowserRouter>
        <Provider store={store}>
          <ThemeProvider theme={theme}>
            <FormProvider {...methods}>
              <ModelBuilderContextProvider>
                <ModelSetup control={methods.control} />
              </ModelBuilderContextProvider>
            </FormProvider>
          </ThemeProvider>
        </Provider>
      </BrowserRouter>
    );
  }

  const targetCols = testModel.config?.dataset?.targetColumns;
  const featureCols = testModel.config?.dataset?.featureColumns;

  if (!targetCols) throw new Error('targetCols is undefined');
  if (!featureCols) throw new Error('featureCols is undefined');

  before(() => {
    cy.on(
      'uncaught:exception',
      (err) => err.toString().includes('ResizeObserver') && false
    );
  });

  beforeEach(() => {
    cy.mount(<MountedComponent />);

    cy.get('#dataset-select')
      .click()
      .type(testModel.config?.dataset?.name || '')
      .get('li[role="option"]')
      .first()
      .click();
  })

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

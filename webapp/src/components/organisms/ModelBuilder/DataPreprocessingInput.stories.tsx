import { ModelCreate } from '@app/rtk/generated/models';
import { store } from '@app/store';
import { schema } from '@features/models/pages/ModelCreateV2';
import { yupResolver } from '@hookform/resolvers/yup';
import { ThemeProvider } from '@mui/system';
import { StoryFn, StoryObj } from '@storybook/react';
import { FormProvider, useForm } from 'react-hook-form';
import { Provider } from 'react-redux';
import { theme } from 'theme';
import DataPreprocessingInput from './DataPreprocessingInput';
import { SimpleColumnConfig } from './types';

export default {
  title: 'components/DataPreprocessingInputt',
  component: DataPreprocessingInput,
  decorators: [
    (Story: StoryFn) => (
      <Provider store={store}>
        <ThemeProvider theme={theme}>
          <Story />
        </ThemeProvider>
      </Provider>
    ),
  ],
  args: {
    value: {
      name: 'v1',
      config: {
        dataset: {
          name: 'Test dataset',
          strategy: 'pipeline',
          featureColumns: [
            {
              name: 'Smiles Column 2',
              dataType: { domainKind: 'smiles' },
              featurizers: [],
              transforms: [],
            },
            {
              name: 'DNA Column 2',
              dataType: { domainKind: 'dna' },
              featurizers: [],
              transforms: [],
            },
            {
              name: 'RNA Column 2',
              dataType: { domainKind: 'rna' },
              featurizers: [],
              transforms: [],
            },
            {
              name: 'Protein Column 2',
              dataType: { domainKind: 'protein' },
              featurizers: [],
              transforms: [],
            },
          ],
          targetColumns: [
            {
              name: 'Numerical Column 1',
              dataType: { domainKind: 'numeric', unit: 'mole' },
              featurizers: [],
              transforms: [],
            },
          ],
        },
        name: 'Test model',
        framework: 'sklearn',
        spec: {
          model: {
            fitArgs: {},
            type: 'sklearn.ensemble.RandomForestRegressor',
          },
        },
      },
    } as ModelCreate,
  },
};

export const SimpleAPI: StoryObj = {
  render: ({ value }: { value?: ModelCreate }) => {
    const methods = useForm<ModelCreate>({
      defaultValues: value,
      mode: 'all',
      criteriaMode: 'all',
      reValidateMode: 'onChange',
      resolver: yupResolver(schema),
    });

    return (
      <>
        <FormProvider {...methods}>
          <DataPreprocessingInput
            value={{
              featureColumns:
                (value?.config?.dataset
                  ?.featureColumns as SimpleColumnConfig[]) || [],
              targetColumns:
                (value?.config?.dataset
                  ?.targetColumns as SimpleColumnConfig[]) || [],
            }}
          />
        </FormProvider>
      </>
    );
  },
};

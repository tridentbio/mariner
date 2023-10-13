import { ModelCreate } from '@app/rtk/generated/models';
import { store } from '@app/store';
import { schema } from '@features/models/pages/ModelForm';
import { yupResolver } from '@hookform/resolvers/yup';
import { ThemeProvider } from '@mui/system';
import { StoryFn, StoryObj } from '@storybook/react';
import { FormProvider, useForm } from 'react-hook-form';
import { Provider } from 'react-redux';
import { theme } from 'theme';
import DataPreprocessingInput from './DataPreprocessingInput';
import { SimpleColumnConfig } from './types';
import { ModelBuilderContextProvider } from './hooks/useModelBuilder';
import { Container } from '@mui/material';

export default {
  title: 'components/DataPreprocessingInput',
  component: DataPreprocessingInput,
  decorators: [
    (Story: StoryFn) => {
      const methods = useForm<ModelCreate>({
        defaultValues: Story.args?.value,
        mode: 'all',
        criteriaMode: 'all',
        reValidateMode: 'onChange',
        resolver: yupResolver(schema),
      });

      return (
        <Provider store={store}>
          <ThemeProvider theme={theme}>
            <ModelBuilderContextProvider>
              <FormProvider {...methods}>
                <Container>
                  <Story />
                </Container>
              </FormProvider>
            </ModelBuilderContextProvider>
          </ThemeProvider>
        </Provider>
      );
    },
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

export const FeatureColumns: StoryObj = {
  render: ({ value }: { value?: ModelCreate }) => {
    return (
      <DataPreprocessingInput
        value={
          (value?.config?.dataset?.featureColumns as SimpleColumnConfig[]) || []
        }
        type="featureColumns"
      />
    );
  },
};

export const TargetColumns: StoryObj = {
  render: ({ value }: { value?: ModelCreate }) => {
    return (
      <DataPreprocessingInput
        value={
          (value?.config?.dataset?.targetColumns as SimpleColumnConfig[]) || []
        }
        type="targetColumns"
      />
    );
  },
};

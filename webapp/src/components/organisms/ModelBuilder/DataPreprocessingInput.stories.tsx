import { store } from '@app/store';
import { yupResolver } from '@hookform/resolvers/yup';
import { ThemeProvider } from '@mui/system';
import { StoryFn, StoryObj } from '@storybook/react';
import { FormProvider, useForm } from 'react-hook-form';
import { Provider } from 'react-redux';
import { theme } from 'theme';
import DataPreprocessingInput from './DataPreprocessingInput';
import { dataPreprocessingFormSchema } from './formSchema';
import { DatasetConfigPreprocessing } from './types';
import { set } from 'cypress/types/lodash';

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
    } as DatasetConfigPreprocessing,
  },
};

export const SimpleAPI: StoryObj = {
  args: {},
  render: (args: { value?: DatasetConfigPreprocessing }) => {
    const methods = useForm<DatasetConfigPreprocessing>({
      defaultValues: args.value,
      mode: 'all',
      criteriaMode: 'all',
      reValidateMode: 'onChange',
      resolver: yupResolver(dataPreprocessingFormSchema),
    });

    const formValue = methods.watch();

    return (
      <>
        <FormProvider {...methods}>
          <DataPreprocessingInput {...args} value={formValue} />
        </FormProvider>
      </>
    );
  },
};

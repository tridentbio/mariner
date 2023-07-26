import { ModelCreate } from '@app/rtk/generated/models';
import { store } from '@app/store';
import { ThemeProvider } from '@mui/material';
import { Meta, StoryObj } from '@storybook/react';
import { FormProvider, useForm } from 'react-hook-form';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { theme } from 'theme';
import { DatasetConfigurationForm } from './DatasetConfigurationForm';

const FormContext = () => {
  const methods = useForm<ModelCreate>({
    mode: 'all',
    defaultValues: {
      name: '',
      modelDescription: '',
      modelVersionDescription: '',
      config: {
        name: '',
        framework: 'torch',
        dataset: {
          featureColumns: [
            {
              name: 'Numerical Column 1',
              dataType: { domainKind: 'numeric', unit: 'mole' },
            },
            { name: 'Smiles Column 2', dataType: { domainKind: 'smiles' } },
            { name: 'DNA Column 2', dataType: { domainKind: 'dna' } },
            { name: 'RNA Column 2', dataType: { domainKind: 'rna' } },
            { name: 'Protein Column 2', dataType: { domainKind: 'protein' } },
          ],
          featurizers: [],
          transforms: [],
          targetColumns: [],
        },
        spec: { layers: [] },
      },
    },
  });

  const { control, getValues, setValue } = methods;

  // // Assert children is a single element and is of type DatasetConfigurationForm
  // if (React.Children.count(children) !== 1 || !React.isValidElement(children) || children.type !== DatasetConfigurationForm) {
  //   console.log(children);
  //   throw new Error(`FormContext must have exactly one child of type DatasetConfigurationForm but got ${children.type}`);
  // }

  // // Pass control and setValue to children
  // React.Children.forEach(children, (child) => {

  //   if (React.isValidElement(child)) {
  //     child = React.cloneElement(child, { control, setValue });
  //   }
  // });

  return (
    <ThemeProvider theme={theme}>
      <Provider store={store}>
        <BrowserRouter>
          <FormProvider {...methods}>
            <form>
              {/* @ts-ignore */}
              <DatasetConfigurationForm control={control} setValue={setValue} />
            </form>
          </FormProvider>
        </BrowserRouter>
      </Provider>
    </ThemeProvider>
  );
};

export default {
  title: 'Dataset Configuration Form',
  component: DatasetConfigurationForm,
  args: {},
  tags: ['forms']
} as Meta;

export const RegularForm = () => <FormContext></FormContext>;

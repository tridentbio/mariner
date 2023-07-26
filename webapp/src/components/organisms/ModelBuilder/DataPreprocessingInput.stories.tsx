import { store } from '@app/store';
import useModelOptions, {
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { StoryObj } from '@storybook/react';
import { useState } from 'react';
import { Provider } from 'react-redux';
import DataPreprocessingInput from './DataPreprocessingInput';

export default {
  title: 'components/DataPreprocessingInputt',
  component: DataPreprocessingInput,
  decorators: [
    (Story) => (
      <Provider store={store}>
        <Story />
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
    },
  },
};

export const SimpleAPI: StoryObj = {
  args: {},
  render: (args) => {
    const [value, setValue] = useState(undefined);

    return (
      <>
        <pre>{JSON.stringify(value, null, 2)}</pre>
        <DataPreprocessingInput {...args} onChange={(val) => setValue(val)} />
      </>
    );
  },
};

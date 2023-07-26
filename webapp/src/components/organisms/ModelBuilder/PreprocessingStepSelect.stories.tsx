import { store } from '@app/store';
import useModelOptions, {
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { StoryObj } from '@storybook/react';
import { useState } from 'react';
import { Provider } from 'react-redux';
import PreprocessingStepSelect from './PreprocessingStepSelect';

export default {
  title: 'components/PreprocessingStepSelect',
  component: PreprocessingStepSelect,
  decorators: [
    (Story) => (
      <Provider store={store}>
        <Story />
      </Provider>
    ),
  ],
  args: {
    options: [
      {
        type: 'sklearn.preprocessing.StandardScaler',
        constructorArgs: {
          with_std: {
            default: true,
            type: 'boolean',
          },
          with_mean: {
            default: true,
            type: 'boolean',
          },
        },
      },
      {
        type: 'fleet.model_builder.featurizers.MoleculeFeaturizer',
        constructorArgs: {
          allow_unknown: {
            default: true,
            type: 'boolean',
          },
          per_atom_fragmentation: {
            default: true,
            type: 'boolean',
          },
          sym_bond_list: {
            default: true,
            type: 'boolean',
          },
        },
      },
    ],
  },
};

export const Simple: StoryObj = {
  args: {},
  render: (args) => {
    const [value, setValue] = useState(undefined);
    return (
      <>
        <PreprocessingStepSelect
          {...args}
          onChange={(val) => console.log(val) || setValue(val)}
        />
        <pre>{JSON.stringify(value, null, 2)}</pre>
      </>
    );
  },
};

export const SimpleAPI: StoryObj = {
  args: {},
  render: (args) => {
    const [value, setValue] = useState(undefined);
    const options = useModelOptions();
    console.log(options);

    if ('error' in options) {
      return <pre>{JSON.stringify(options.error, null, 2)}</pre>;
    }

    const preprocessingOptions = options.options.map(toConstructorArgsConfig);

    const option = options.options.find((o) => o.classPath === value?.type);

    console.log(options);
    console.log(option);

    return (
      <>
        <pre>{JSON.stringify(value, null, 2)}</pre>
        <PreprocessingStepSelect
          {...args}
          onChange={(val) => setValue(val)}
          options={preprocessingOptions}
        />
        <pre>{JSON.stringify(option, null, 2)}</pre>
      </>
    );
  },
};

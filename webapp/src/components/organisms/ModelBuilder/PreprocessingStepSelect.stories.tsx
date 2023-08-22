import { store } from '@app/store';
import useModelOptions, {
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { StoryFn, StoryObj } from '@storybook/react';
import { useState } from 'react';
import { Provider } from 'react-redux';
import PreprocessingStepSelect from './PreprocessingStepSelect';
import { GenericPreprocessingStep, StepValue } from './types';

export default {
  title: 'components/PreprocessingStepSelect',
  component: PreprocessingStepSelect,
  decorators: [
    (Story: StoryFn) => (
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
  render: (args: { [key: string]: any }) => {
    const [value, setValue] = useState<GenericPreprocessingStep | null>(null);
    return (
      <>
        <PreprocessingStepSelect
          options={args.options as StepValue[]}
          onChanges={(val) => setValue(val)}
        />
        <pre>{JSON.stringify(value, null, 2)}</pre>
      </>
    );
  },
};

export const SimpleAPI: StoryObj = {
  args: {},
  render: (args) => {
    const [value, setValue] = useState<any>(undefined);
    const options = useModelOptions();

    /* if ('error' in options) {
      return <pre>{JSON.stringify(options.error, null, 2)}</pre>;
    } */

    const preprocessingOptions = options.options.map(toConstructorArgsConfig);

    const option = options.options.find((o) => o.classPath === value?.type);

    return (
      <>
        <pre>{JSON.stringify(value, null, 2)}</pre>
        <PreprocessingStepSelect
          {...args}
          onChanges={(val) => setValue(val)}
          options={preprocessingOptions as StepValue[]}
        />
        <pre>{JSON.stringify(option, null, 2)}</pre>
      </>
    );
  },
};

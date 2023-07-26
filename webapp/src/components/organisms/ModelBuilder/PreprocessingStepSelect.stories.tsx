import { StoryObj } from '@storybook/react';
import { useState } from 'react';
import PreprocessingStepSelect from './PreprocessingStepSelect';

export default {
  title: 'components/PreprocessingStepSelect',
  component: PreprocessingStepSelect,
  args: {
    options: [
      {
        type: 'sklearn.preprocessing.StandardScaler',
        constructorArgs: {
          with_std: {
            default: true,
            type: 'bool',
          },
          with_mean: {
            default: true,
            type: 'bool',
          },
        },
      },
      {
        type: 'fleet.model_builder.featurizers.MoleculeFeaturizer',
        constructorArgs: {
          allow_unknown: {
            default: true,
            type: 'bool',
          },
          per_atom_fragmentation: {
            default: true,
            type: 'bool',
          },
          sym_bond_list: {
            default: true,
            type: 'bool',
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

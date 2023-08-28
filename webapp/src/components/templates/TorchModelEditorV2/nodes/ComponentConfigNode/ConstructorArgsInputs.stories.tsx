import { StoryObj } from '@storybook/react';
import ConstructorArgsInputs, {
  ConstructorArgsInputsProps,
} from './ConstructorArgsInputs';

const defaultArgs: ConstructorArgsInputsProps = {
  editable: true,
  data: {
    name: 'feat',
    type: 'fleet.model_builder.featurizers.IntegerFeaturizer',
    forwardArgs: {
      input_: '$dna',
    },
  },
};

export default {
  title: 'components/ConstructorArgsInputs',
  component: ConstructorArgsInputs,
  args: defaultArgs,
};

export const A: StoryObj = {};

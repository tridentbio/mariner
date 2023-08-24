import { StoryObj } from '@storybook/react';
import {
  TransformConstructorForm,
  TransformerConstructorFormProps,
} from './TransformConstructorForm';

const defaultArgs = {
  transformer: {
    name: 'std-scaler',
    constructorArgs: {
      with_mean: true,
      with_std: false,
    },
    fowardArgs: {},
    type: 'sklearn.preprocessing.StandardScaler',
  },
};

export default {
  title: 'TransformConstructorForm',
  component: TransformConstructorForm,
  args: defaultArgs,
};

export const RegularTransformerConstructorForm: StoryObj = {};

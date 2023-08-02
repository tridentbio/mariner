
import { store } from '@app/store';
import useModelOptions, {
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { StoryObj } from '@storybook/react';
import { useState } from 'react';
import { Provider } from 'react-redux';
import PreprocessingStepSelect from './PreprocessingStepSelect';
import SklearnModelInput from './SklearnModelInput';

export default {
  title: 'components/SklearnModelInput',
  component: SklearnModelInput,
  decorators: [
    (Story) => (
      <Provider store={store}>
        <Story />
      </Provider>
    ),
  ],
  args: {
  }
};

export const Simple: StoryObj = {
  args: {},
  render: (args) => {
    const [value, setValue] = useState(undefined);
    return (
      <>
        <SklearnModelInput/>
        <pre>{JSON.stringify(value, null, 2)}</pre>
      </>
    );
  },
};


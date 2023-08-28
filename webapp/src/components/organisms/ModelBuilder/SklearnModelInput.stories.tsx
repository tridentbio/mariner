import { ModelCreate } from '@app/rtk/generated/models';
import { store } from '@app/store';
import { schema } from '@features/models/pages/ModelCreateV2';
import { yupResolver } from '@hookform/resolvers/yup';
import { Container, ThemeProvider } from '@mui/material';
import { StoryFn, StoryObj } from '@storybook/react';
import { FormProvider, useForm } from 'react-hook-form';
import { Provider } from 'react-redux';
import { theme } from 'theme';
import SklearnModelInput from './SklearnModelInput';
import { ModelBuilderContextProvider } from './hooks/useModelBuilder';

export default {
  title: 'components/SklearnModelInput',
  component: SklearnModelInput,
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
  args: {},
};

export const Simple: StoryObj = {
  args: {},
  render: (args) => {
    return <SklearnModelInput />;
  },
};

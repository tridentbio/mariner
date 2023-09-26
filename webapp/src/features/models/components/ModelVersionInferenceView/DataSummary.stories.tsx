import { store } from '@app/store';
import { ThemeProvider } from '@mui/material';
import { StoryFn, StoryObj } from '@storybook/react';
import { Provider } from 'react-redux';
import { theme } from 'theme';
import { getDataset } from '../../../../../tests/fixtures/datasets';
import DataSummary, { DataSummaryProps } from './DataSummary';

export default {
  title: 'components/DataSummary',
  component: DataSummary,
  decorators: [
    (Story: StoryFn) => {
      return (
        <Provider store={store}>
          <ThemeProvider theme={theme}>
            <Story />
          </ThemeProvider>
        </Provider>
      );
    },
  ],
  args: {
    value: {
      titlePrefix: 'Data Summary',
      inference: [
        { columnName: 'mwt', value: 500 },
        { columnName: 'tpsa', value: 500 },
      ],
      columnsData: getDataset().stats,
    },
  },
};

export const A: StoryObj<{ value: DataSummaryProps }> = {
  render: (args: { value: DataSummaryProps }) => {
    const { value } = args;
    return <DataSummary {...value} />;
  },
};

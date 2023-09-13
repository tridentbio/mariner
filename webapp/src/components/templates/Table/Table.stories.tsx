import { store } from '@app/store';
import { ThemeProvider } from '@mui/material';
import { StoryFn, StoryObj } from '@storybook/react';
import { Provider } from 'react-redux';
import { theme } from 'theme';
import Table, { Column } from '.';
import {
  columns,
  rows,
} from '../../../../tests/fixtures/table/experimentsDataMock';

export default {
  title: 'components/Table',
  component: Table,
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
      columns,
    },
  },
};

export const ExperimentsTable: StoryObj = {
  render: ({ value }: { value?: { columns: Column<any, any>[] } }) => {
    if (!value) return <div>error</div>;

    return (
      <Table
        columns={value.columns}
        rows={rows}
        rowKey={() => 'random'}
        filterLinkOperatorOptions={['and', 'or']}
      />
    );
  },
};

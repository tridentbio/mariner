import { Button } from '@mui/material';
import Content from 'components/templates/AppLayout/Content';
import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Table from 'components/templates/Table';
import {
  Experiment,
  FetchExperimentsQuery,
} from '@app/types/domain/experiments';
import { trainingListingColumns } from './columns';

import { experimentsApi } from '@app/rtk/experiments';
import { State } from 'components/templates/Table/types';

const TraininingListing = () => {
  const navigate = useNavigate();
  const [queryParams, setQueryParams] = useState<FetchExperimentsQuery>({
    perPage: 10,
    page: 0,
    orderBy: '-createdAt',
  });
  const { data: paginatedExperiments } = experimentsApi.useGetExperimentsQuery({
    ...queryParams,
  });
  const { total, experiments } = useMemo(() => {
    return {
      experiments: paginatedExperiments?.data || [],
      total: paginatedExperiments?.total || 0,
    };
  }, [paginatedExperiments]);

  const handleTableStateChange = (state: State) => {
    const newQueryParams: FetchExperimentsQuery = {};
    if (state.paginationModel) {
      const { page, rowsPerPage: perPage } = state.paginationModel;
      newQueryParams.page = page;
      newQueryParams.perPage = perPage;
    }
    if (state.filterModel.items.length) {
      newQueryParams.stage = state.filterModel.items.map((item) =>
        (item.value as string).toUpperCase()
      );
    }
    if (state.sortModel.length) {
      newQueryParams.orderBy = state.sortModel.reduce((acc, item, index) => {
        const signal = item.sort === 'asc' ? '+' : '-';
        if (!index) {
          return `${signal}${item.field}`;
        }
        return `${acc},${signal}${item.field}`;
      }, '');
    }
    setQueryParams((prev) => ({ ...prev, ...newQueryParams }));
  };

  return (
    <Content>
      <Button
        variant="contained"
        color="primary"
        onClick={() => navigate(`/trainings/new`)}
        id="go-to-create-training"
        sx={{ my: 1, float: 'right' }}
      >
        Create Training
      </Button>

      <Table<Experiment>
        onStateChange={(state) => handleTableStateChange(state)}
        rows={experiments}
        columns={trainingListingColumns}
        rowKey={(row) => row.id}
        rowAlign="center"
        pagination={{
          total,
          page: queryParams.page || 0,
          rowsPerPage: queryParams.perPage || 10,
        }}
      />
    </Content>
  );
};

export default TraininingListing;

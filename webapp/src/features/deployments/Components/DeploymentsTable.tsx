import { useEffect, useState } from 'react';
import Table from 'components/templates/Table';
import { Column } from 'components/templates/Table/types';
import { dateRender } from 'components/atoms/Table/render';
import { deploymentsApi } from '../deploymentsApi';
import { Deployment } from '../types';
import StatusChip from './StatutsChip';
import DeploymentsTableActions from './TableActions';
import { Box, Chip } from '@mui/material';
import { useAppDispatch } from '@hooks';
import { setCurrentDeployment } from '../deploymentsSlice';

interface DeploymentsTableProps {
  toggleModal: () => void;
  handleClickDelete: (id: number) => void;
}

const DeploymentsTable: React.FC<DeploymentsTableProps> = ({
  toggleModal,
  handleClickDelete,
}) => {
  const [getDeployments, { isLoading, data, originalArgs }] =
    deploymentsApi.useLazyGetDeploymentsQuery();
  const dispatch = useAppDispatch();
  useEffect(() => {
    getDeployments({
      page: 0,
      perPage: 10,
    });
  }, []);

  const columns: Column<Deployment, any>[] = [
    {
      field: 'name',
      title: 'Deployment Name',
      name: 'Deployment Name',
      skeletonProps: {
        variant: 'text',
        width: 60,
      },
      render: (row) => row.name,
    },
    {
      field: 'modelVersion',
      name: 'Model Version',
      title: 'Model Version',
      skeletonProps: {
        variant: 'rectangular',
        height: 50,
      },
      render: (row) => row.modelVersion?.name,
    },
    {
      field: 'status',
      title: 'Status',
      name: 'Status',
      skeletonProps: {
        variant: 'rectangular',
        width: 60,
      },
      render: (row) => (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <StatusChip status={row.status} />
        </Box>
      ),
    },
    {
      field: 'displayTrainingData',
      title: 'Display Training Data',
      name: 'Display Training Data',
      skeletonProps: {
        variant: 'rectangular',
        width: 20,
      },
      render: (row) => (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Chip
            label={row.showTrainingData ? 'YES' : 'NO'}
            color={row.showTrainingData ? 'success' : 'error'}
          />
        </Box>
      ),
    },
    {
      field: 'shareStrategy',
      title: 'Share Strategy',
      name: 'Share Strategy',
      skeletonProps: {
        variant: 'circular',
        width: 30,
        height: 30,
      },
      render: (row) => row.shareStrategy,
    },
    {
      field: 'rateLimit',
      title: 'Rate Limit',
      name: 'Rate Limit',
      skeletonProps: {
        variant: 'rectangular',
        width: 30,
      },
      render: (row) => (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {`${row.predictionRateLimitValue}/${row.predictionRateLimitUnit}`}
        </Box>
      ),
    },
    {
      field: 'createdAt',
      title: 'Created At',
      name: 'Created At',
      skeletonProps: {
        variant: 'circular',
        width: 30,
        height: 30,
      },
      render: dateRender((row) => row.createdAt),
    },
    {
      name: 'Action',
      field: 'Actions',
      title: 'Actions',
      customSx: {
        position: 'sticky',
        right: -1,
        background: '#f3f3f3',
        textAlign: 'center',
      },
      bold: true,
      render: (row) => (
        <DeploymentsTableActions
          onClickDelete={handleClickDelete}
          onClickEdit={() => {
            toggleModal();
            dispatch(setCurrentDeployment(row));
          }}
          id={row.id}
          status={row.status}
        />
      ),
    },
  ];

  return (
    <div style={{ width: '100%', overflowX: 'auto', display: 'block' }}>
      <Table<Deployment>
        loading={isLoading}
        rowKey={(row) => row.name}
        rows={data?.data || []}
        onStateChange={(state) => {
          getDeployments({
            page: state.paginationModel?.page || 0,
            perPage: state.paginationModel?.rowsPerPage || 10,
          });
        }}
        pagination={{
          total: data?.total || 0,
          page: originalArgs?.page || 0,
          rowsPerPage: originalArgs?.perPage || 0,
        }}
        columns={columns}
        extraTableStyle={{ marginBottom: 0 }}
      />
    </div>
  );
};

export default DeploymentsTable;

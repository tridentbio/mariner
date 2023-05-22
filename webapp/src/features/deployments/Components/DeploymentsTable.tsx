import { useEffect, useState } from 'react';
import Table from 'components/templates/Table';
import { Column } from 'components/templates/Table/types';
import { dateRender } from 'components/atoms/Table/render';
import * as deploymentsApi from 'app/rtk/generated/deployments';
import { Deployment } from 'app/rtk/generated/deployments';
import StatusChip from './StatutsChip';
import DeploymentsTableActions from './TableActions';
import { Box, Chip, Link, Tab, Tabs } from '@mui/material';
import { useAppDispatch } from 'app/hooks';
import { setCurrentDeployment } from '../deploymentsSlice';
import { linkRender } from 'components/atoms/Table/render';
import { GetDeploymentsApiArg } from 'app/rtk/generated/deployments';
import { useAppSelector } from '@hooks';

interface DeploymentsTableProps {
  toggleModal?: () => void;
  handleClickDelete?: (id: number) => void;
  fixedTab?: number;
}

const TabOptions: {
  name: string;
  filter: GetDeploymentsApiArg;
}[] = [
  {
    name: 'All',
    filter: {
      publicMode: 'include',
    },
  },
  {
    name: 'Public',
    filter: {
      publicMode: 'only',
    },
  },
  {
    name: 'Shared',
    filter: {
      accessMode: 'shared',
    },
  },
  {
    name: 'My',
    filter: {
      accessMode: 'owned',
    },
  },
];

const DeploymentsTable: React.FC<DeploymentsTableProps> = ({
  toggleModal,
  handleClickDelete,
  fixedTab,
}) => {
  const [option, setOption] = useState(fixedTab || 0);
  const [getDeployments, { isLoading, originalArgs, data }] =
    deploymentsApi.useLazyGetDeploymentsQuery();
  const { deployments } = useAppSelector((state) => state.deployments);
  const dispatch = useAppDispatch();
  useEffect(() => {
    const optionChosed = TabOptions[option];
    getDeployments({
      page: 0,
      perPage: 10,
      ...optionChosed.filter,
    });
  }, [option]);

  const columns: Column<Deployment, any>[] = [
    {
      field: 'name',
      title: 'Deployment Name',
      name: 'Deployment Name',
      skeletonProps: {
        variant: 'text',
        width: 60,
      },
      render: linkRender(
        (row: Deployment) => `/deployments/${row.id}`,
        (row: Deployment) => row.name
      ),
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
      render: (row) => {
        if (!row.createdAt) return '';
        return dateRender<typeof row>((row) => new Date(row.createdAt!))(row);
      },
    },
  ];

  option === 3 &&
    columns.push({
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
      render: (row) =>
        row.createdById === 1 && (
          <DeploymentsTableActions
            onClickDelete={handleClickDelete}
            onClickEdit={
              toggleModal &&
              (() => {
                toggleModal();
                dispatch(setCurrentDeployment(row));
              })
            }
            id={row.id}
            status={row.status}
            shareUrl={row.shareUrl}
          />
        ),
    });

  return (
    <div style={{ width: '100%', overflowX: 'auto', display: 'block' }}>
      {fixedTab === undefined && (
        <Tabs value={option} onChange={(_, v) => setOption(v)}>
          {TabOptions.map((tab, index) => (
            <Tab key={tab.name} label={tab.name} value={index} />
          ))}
        </Tabs>
      )}
      <Table<Deployment>
        loading={isLoading}
        rowKey={(row) => row.name}
        rows={deployments || []}
        onStateChange={(state) => {
          getDeployments({
            page: state.paginationModel?.page || 0,
            perPage: state.paginationModel?.rowsPerPage || 10,
            ...TabOptions[option].filter,
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

import { Avatar, IconButton } from '@mui/material';
import ReadMoreIcon from '@mui/icons-material/ReadMore';

import { useEffect } from 'react';
import { modelsApi } from 'app/rtk/models';
import { Model } from 'app/types/domain/models';
import Table from 'components/templates/Table';
import { Column } from 'components/templates/Table/types';
import { linkRender } from 'components/atoms/Table/render';
import {
  TableActionsWrapper,
  tableActionsSx,
} from '@components/atoms/TableActions';

interface IModelTable {
  onOpenDetails: (model: Model) => void;
}

const ModelTable = (props: IModelTable) => {
  const [getModels, { isLoading, data, originalArgs }] =
    modelsApi.useLazyGetModelsOldQuery();
  const TableActions = ({ onOpenDetails }: { onOpenDetails: () => any }) => (
    <>
      <IconButton onClick={onOpenDetails}>
        <ReadMoreIcon />
      </IconButton>
    </>
  );
  useEffect(() => {
    getModels({
      page: 0,
      perPage: 10,
    });
  }, []);

  const columns: Column<Model, any>[] = [
    {
      field: 'name',
      title: 'Name',
      name: 'Name',
      skeletonProps: {
        variant: 'text',
        width: 60,
      },
      render: linkRender(
        (row: Model) => `/models/${row.id}`,
        (row: Model) => row.name
      ),
    },
    {
      field: 'description',
      name: 'Description',
      title: 'Description',
      skeletonProps: {
        variant: 'rectangular',
        height: 50,
      },
    },
    {
      field: null,
      render: (row) => row.versions[row.versions.length - 1]?.name,
      title: 'Latest Version',
      name: 'Latest Version',
      skeletonProps: {
        variant: 'text',
        width: 60,
      },
    },
    {
      field: 'createdBy',
      title: 'Created By',
      name: 'Created By',
      render: (_, value) => <Avatar src={value?.avatarUrl} />,
      skeletonProps: {
        variant: 'circular',
        width: 30,
        height: 30,
      },
    },
    {
      name: 'Action',
      field: null,
      title: 'Actions',
      customSx: tableActionsSx,
      render: (row) => (
        <TableActionsWrapper>
          <TableActions onOpenDetails={() => props.onOpenDetails(row)} />
        </TableActionsWrapper>
      ),
    },
  ];

  return (
    <div style={{ height: 800 }}>
      <Table<Model>
        loading={isLoading}
        rowKey={(row) => row.name}
        rows={data?.data || []}
        onStateChange={(state) => {
          getModels({
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
      />
    </div>
  );
};

export default ModelTable;

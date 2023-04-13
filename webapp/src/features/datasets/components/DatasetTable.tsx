import { Avatar, IconButton } from '@mui/material';
import ReadMoreIcon from '@mui/icons-material/ReadMore';
import DownloadIcon from '@mui/icons-material/Download';
import { downloadDataset } from '../datasetSlice';
import { useAppSelector } from '@hooks';
import { useEffect } from 'react';
import Table from 'components/templates/Table';
import { Column, State } from 'components/templates/Table/types';
import AppLink from 'components/atoms/AppLink';
import { dateRender } from 'components/atoms/Table/render';
import { datasetsApi } from '@app/rtk/datasets';
import { Dataset } from '@app/types/domain/datasets';
import { StatusButton } from './StatusButton';
import Justify from 'components/atoms/Justify';

interface DatasetTableProps {
  onOpenDetails?: (datasetId: number) => void;
}
const DatasetTable = (props: DatasetTableProps) => {
  const { datasets, total, filters } = useAppSelector(
    (state) => state.datasets
  );
  const [fetchDatasets, { isLoading }] = datasetsApi.useLazyGetDatasetsQuery();

  useEffect(() => {
    if (!isLoading) fetchDatasets(filters);
  }, [filters]);

  const TableActions = ({
    onOpenDetails,
    onDownload,
  }: {
    onOpenDetails: () => any;
    onDownload: () => any;
  }) => (
    <>
      <IconButton onClick={onOpenDetails}>
        <ReadMoreIcon />
      </IconButton>

      <IconButton onClick={onDownload}>
        <DownloadIcon />
      </IconButton>
    </>
  );

  const columns: Column<Dataset, any>[] = [
    {
      field: 'name',
      render: (row, val) => (
        <Justify position="start">
          <AppLink to={`/datasets/${row.id}`}>{val}</AppLink>
        </Justify>
      ),
      title: 'NAME',
      filterSchema: {
        byValue: true,
      },
      name: 'Name',
    },
    {
      field: 'rows',
      title: 'ROWS',
      render: (_row, val) => <Justify position="end">{val}</Justify>,
      sortable: true,
      name: 'Rows',
      customSx: {
        textAlign: 'center',
      },
    },
    {
      field: 'columns',
      title: 'COLUMNS',
      render: (_row, val) => <Justify position="end">{val}</Justify>,
      sortable: true,
      name: 'Columns',
      customSx: {
        textAlign: 'center',
      },
    },
    {
      field: 'createdAt',
      title: 'CREATED AT',
      render: (row) => (
        <Justify position="start">
          {dateRender((thing: Dataset) => thing.createdAt)(row)}
        </Justify>
      ),
      sortable: true,
      name: 'Created At',
    },
    {
      field: 'createdBy',
      title: 'CREATED BY',
      render: (_, value) => (
        <Justify position="center">
          <Avatar src={value?.avatarUrl} />
        </Justify>
      ),
      name: 'Created By',
      customSx: {
        textAlign: 'center',
      },
    },
    {
      field: 'readyStatus',
      title: 'STATUS',
      render: (_, value) => (
        <Justify position="center">
          <StatusButton status={value || ''} />
        </Justify>
      ),
      name: 'Ready Status',
      customSx: {
        textAlign: 'center',
      },
    },
    {
      field: null,
      title: '',
      render: (row) => (
        <Justify position="center">
          <TableActions
            onOpenDetails={() =>
              props.onOpenDetails && props.onOpenDetails(row.id)
            }
            onDownload={() => downloadDataset(row.id, row.dataUrl)}
          />
        </Justify>
      ),
      name: 'Actions',
    },
  ];

  const handleTableStateChange = (state: State) => {
    const pFilters: Partial<typeof filters> = {};
    const searchByFilter = state.filterModel.items.find(
      (item) => item.columnName === 'Name' && item.operatorValue === 'eq'
    );

    if (searchByFilter) {
      pFilters.searchByName = searchByFilter.value;
    }
    const rowsSort = state.sortModel.find((item) => item.field === 'rows');
    const colsSort = state.sortModel.find((item) => item.field === 'columns');
    const createdAtSort = state.sortModel.find(
      (item) => item.field === 'createdAt'
    );

    if (rowsSort) pFilters.sortByRows = rowsSort.sort;
    if (colsSort) pFilters.sortByCols = colsSort.sort;
    if (createdAtSort) pFilters.sortByCreatedAt = createdAtSort.sort;
    const newFilters: Parameters<typeof fetchDatasets>[0] = {
      page: state.paginationModel?.page || 0,
      perPage: state.paginationModel?.rowsPerPage || filters.perPage,
      ...pFilters,
    };

    fetchDatasets(newFilters);
  };

  return (
    <div
      style={{
        overflowX: 'scroll',
        width: '100%',
        overflowY: 'clip',
      }}
    >
      <Table<Dataset>
        loading={isLoading}
        onStateChange={handleTableStateChange}
        rows={datasets}
        columns={columns}
        filterLinkOperatorOptions={['or', 'and']}
        rowKey={(row) => row.id}
        pagination={{
          total: total,
          rowsPerPage: filters?.perPage || 25,
          page: filters?.page || 0,
        }}
      />
    </div>
  );
};

export default DatasetTable;

import { Column, TableProps } from '@components/templates/Table';
import { State } from '@components/templates/Table/types';
import { TablePaginationProps } from '@mui/material';
import { NonUndefined, deepClone } from '@utils';
import { createContext, useMemo, useState } from 'react';
import { filterRows } from './filterValidation';

export const useTableFilters = <R extends { [key: string]: any }>({
  columns,
  rows,
  pagination,
  dependencies = {},
  linkOperator = 'and',
}: {
  columns: Column<any, any>[];
  rows: TableProps<R>['rows'];
  pagination: TableProps<R>['pagination'];
  dependencies: TableProps<R>['dependencies'];
  linkOperator?: NonUndefined<TableProps<R>['filterLinkOperatorOptions']>[0];
}) => {
  const [filters, setFilters] = useState<State>({
    filterModel: { items: [], linkOperator },
    sortModel: [],
    paginationModel: pagination,
  });

  const filterableColumns = useMemo(
    () => columns.filter((col) => !!col.filterSchema),
    [columns]
  );

  const sortRows = (rowsList: R[]) => {
    return rowsList.sort((a, b) => {
      for (let sort of filters.sortModel) {
        const columnRef = columns.find((col) => col.field === sort.field);
        const aValue =
          columnRef?.valueGetter?.(a, dependencies) || a[sort.field];
        const bValue =
          columnRef?.valueGetter?.(b, dependencies) || b[sort.field];

        if (aValue > bValue) return sort.sort == 'asc' ? 1 : -1;
        if (aValue < bValue) return sort.sort == 'asc' ? -1 : 1;
      }

      return 0;
    });
  };

  const filteredRows = useMemo(() => {
    let data = deepClone(
      filterRows(filters.filterModel, columns, rows, dependencies)
    ) as R[];

    if (filters.sortModel.length) {
      data = sortRows(data);
    }

    return data;
  }, [rows, filters]);

  const handlePageChange: TablePaginationProps['onPageChange'] = (
    _event,
    newPage
  ) => {
    if (!filters.paginationModel) {
      return;
    }
    const newState: State = {
      ...filters,
      paginationModel: { ...filters.paginationModel, page: newPage },
    };
    setFilters(newState);
  };

  const handleRowsPerPageChange: TablePaginationProps['onRowsPerPageChange'] = (
    _event
  ) => {
    if (!filters.paginationModel) return;

    const newRowsPerPage = parseInt(_event.target.value);

    setFilters((prev) => ({
      ...prev,
      paginationModel: {
        ...(prev.paginationModel || { page: 0 }),
        rowsPerPage: newRowsPerPage,
        total: prev.paginationModel?.total || 0,
      },
    }));
  };

  const getColumnState = (col: Column<any, any>) => {
    return {
      filters: filters.filterModel.items.filter(
        (item) => item.columnName === col.field
      ),
      sort: filters.sortModel.find((item) => item.field === col.field),
    };
  };

  return {
    filterableColumns,
    filteredRows,
    filters,
    setFilters,
    handlePageChange,
    handleRowsPerPageChange,
    getColumnState,
  };
};

export interface TableFiltersContextProps {
  filters: State;
  setFilters: ReturnType<typeof useTableFilters>['setFilters'];
  filterableColumns: ReturnType<typeof useTableFilters>['filterableColumns'];
}

//@ts-ignore
export const TableFilterContext = createContext<TableFiltersContextProps>({});

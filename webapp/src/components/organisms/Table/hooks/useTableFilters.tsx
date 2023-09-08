import { Column, TableProps } from '@components/templates/Table';
import { State } from '@components/templates/Table/types';
import { TablePaginationProps } from '@mui/material';
import {
  ReactNode,
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';

interface TableContextProps {
  editable: boolean;
  setEditable: React.Dispatch<React.SetStateAction<boolean>>;
  defaultExpanded: boolean;
}

// @ts-ignore
/* const TableContext = createContext<TableContextProps>({});

export const TableContextProvider = () => {
  const [isEditable, setIsEditable] = useState<boolean>(editable);

  return (
    <TableContext.Provider
      value={{
        editable: isEditable,
        setEditable: setIsEditable,
        defaultExpanded,
      }}
    >
      {children}
    </TableContext.Provider>
  );
};
 */
const useTableFilters = <R extends { [key: string]: any }>({
  columns,
  rows,
  pagination,
}: {
  columns: Column<any, any>[];
  rows: TableProps<R>['rows'];
  pagination: TableProps<R>['pagination'];
}) => {
  const [filters, setFilters] = useState<State>({
    filterModel: { items: [] },
    sortModel: [],
    paginationModel: pagination,
  });

  const filterableColumns = useMemo(
    () => columns.filter((col) => !!col.filterSchema),
    [columns]
  );

  const filteredRows = useMemo(() => rows, [rows]);

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
    /* const newState: State = {
      ...filters,
      paginationModel: {
        ...(filters.paginationModel || { page: 0 }),
        rowsPerPage: newRowsPerPage,
      },
    };

    setFilters(newState); */
  };

  const getColumnState = (col: Column<any, any>) => {
    return {
      filters: filters.filterModel.items.filter(
        (item) => item.columnName === col.field
      ),
      sort: filters.sortModel.find((item) => item.field === col.field),
    };
  };

  useEffect(() => {}, [filters]);

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

export default useTableFilters;

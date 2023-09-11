import {
  Box,
  Table as MuiTable,
  Skeleton,
  TableBody,
  TableCell,
  TableFooter,
  TableHead,
  TablePagination,
  TableRow,
  Typography,
} from '@mui/material';
import { ReactNode, useEffect, useMemo, useRef, useState } from 'react';

import { useAppSelector } from '@app/hooks';
import { DraggableCell } from '@components/organisms/Table/DraggableCell';
import { SortableRow } from '@components/organisms/Table/SortableRow';
import {
  TableFilterContext,
  TableFiltersContextProps,
  useTableFilters,
} from '@components/organisms/Table/hooks/useTableFilters';
import { setPreference } from '@features/users/usersSlice';
import { useAppDispatch } from '@hooks';
import NoData from 'components/atoms/NoData';
import Filters, { FilterProps } from 'components/organisms/Table/Filters';
import { range } from 'utils';
import FilterIndicator from './FilterIndicator';
import SortingButton from './SortingButton';
import SortingIndicator from './SortingIndicator';
import { colTitle, columnId, isColumnSortable } from './common';
import { Column, TableProps } from './types';

const isKeyOf = <O,>(
  key: string | number | symbol | null,
  obj: O extends object ? O : never
): key is keyof O => {
  if (!key) return false;
  return key in obj;
};

const TableFiltersContextProvider = ({
  children,
  filterableColumns,
  filters,
  setFilters,
}: TableFiltersContextProps & { children: ReactNode }) => {
  const params = useMemo<TableFiltersContextProps>(
    () => ({
      filters,
      setFilters,
      filterableColumns,
    }),
    [setFilters, filters]
  );

  return (
    <TableFilterContext.Provider value={params}>
      {children}
    </TableFilterContext.Provider>
  );
};

const Table = <R extends { [key: string]: any }>({
  columns,
  rows,
  rowKey,
  filterLinkOperatorOptions,
  pagination,
  onStateChange,
  noData,
  loading,
  rowAlign,
  rowCellStyle,
  extraTableStyle,
  usePreferences,
  tableId,
  dependencies = {},
  columnTree,
}: TableProps<R>) => {
  const preferences = useAppSelector((state) => state.users.preferences);
  const preferencesLoaded = useRef<boolean>(false);
  const dispatch = useAppDispatch();

  const [allColumns, setAllColumns] = useState<Column<any, any>[]>(columns);

  const displayedColumns = useMemo<Column<any, any>[]>(() => {
    return allColumns.filter((col) => !col.hidden || col.fixed);
  }, [allColumns]);

  const {
    filterableColumns,
    filters,
    filteredRows,
    setFilters,
    handlePageChange,
    handleRowsPerPageChange,
    getColumnState,
  } = useTableFilters({
    columns: allColumns,
    rows,
    pagination,
    dependencies,
  });

  const renderCol = (row: any, { render, field }: Column<any, any>) => {
    if (isKeyOf(field, row)) {
      return render ? render(row, row[field], dependencies) : row[field];
    } else if (render) {
      return render(row, null, dependencies);
    } else {
      throw new Error(`Should have valid field or render. Either "${field}"\n
      is not a key of row, or there is no render`);
    }
  };

  const cellStyle = useMemo<TableProps<R>['rowCellStyle']>(() => {
    if (!rowCellStyle) return {};
    if (rowAlign === 'center') {
      return {
        textAlign: 'center',
        verticalAlign: 'middle',
      };
    }
  }, [rowCellStyle]);

  useEffect(() => {
    if (onStateChange) {
      onStateChange(filters);
    }
  }, [filters]);

  useEffect(() => {
    if (!usePreferences || !tableId || preferencesLoaded.current) return;

    preferences.tables && onLoadedPreferences();
  }, [preferences.tables]);

  const onLoadedPreferences = () => {
    if (preferences.tables && tableId && preferences.tables[tableId]) {
      const tablePreferences = preferences.tables;

      if (tablePreferences) {
        sortColumnPositions(
          tablePreferences[tableId]?.columns.map((col) => col.name) || []
        );

        preferencesLoaded.current = true;
      }
    }
  };

  const updateTablePreferences = (data: any) => {
    if (usePreferences && tableId) {
      dispatch(
        setPreference({
          path: `tables.${tableId}`,
          data,
        })
      );
    }
  };

  const sortColumnPositions = (sortedAndDisplayedColIds: string[]) => {
    if (!sortedAndDisplayedColIds.length) return;

    const colsToDisplay: Column<any, any>[] = [];

    sortedAndDisplayedColIds.forEach((sortedColId) => {
      const foundCol = allColumns.find((col) => sortedColId === col.name);

      if (foundCol) colsToDisplay.push({ ...foundCol, hidden: false });
    });

    const hiddenCols: Column<any, any>[] = allColumns
      .filter((col) => !colsToDisplay.some((c) => c.name == col.name))
      .map((col) => ({ ...col, hidden: true }));

    setAllColumns([...colsToDisplay, ...hiddenCols]);
  };

  const onDroppedColumn = (sortedAndDisplayedCols: Column<any, any>[]) => {
    sortColumnPositions(sortedAndDisplayedCols.map((col) => col.name));

    updateTablePreferences({
      columns: sortedAndDisplayedCols.map((col) => ({ name: col.name })),
    });
  };

  const handleSelectedColumns: FilterProps['onSelectedColumns'] = (
    selectedColumnsIdList
  ) => {
    setAllColumns((prev) => {
      return prev.map((col) => {
        col.hidden = !selectedColumnsIdList.includes(col.name as string);

        return col;
      });
    });

    updateTablePreferences({
      columns: allColumns
        .filter((col) => selectedColumnsIdList.includes(col.name))
        .map((col) => ({ name: col.name })),
    });
  };

  return (
    <Box
      sx={{
        width: '100%',
        overflowX: 'auto',
        display: 'block',
      }}
    >
      <MuiTable
        sx={{
          border: '1px solid rgb(224, 224, 224)',
          mb: 6,
          ...extraTableStyle,
        }}
      >
        <TableFiltersContextProvider
          filterableColumns={filterableColumns}
          filters={filters}
          setFilters={setFilters}
        >
          <TableHead>
            {(!!filterableColumns.length || !!filters.sortModel) && (
              <TableRow>
                <TableCell sx={{ padding: 1 }} colSpan={24}>
                  <Filters
                    filterLinkOperatorOptions={filterLinkOperatorOptions}
                    columns={allColumns}
                    onSelectedColumns={handleSelectedColumns}
                    treeView={columnTree}
                  />
                </TableCell>
              </TableRow>
            )}
            <SortableRow columns={displayedColumns} onDropped={onDroppedColumn}>
              {displayedColumns.map((col, index) => {
                const { filters: colFilters, sort } = getColumnState(col);

                return (
                  <DraggableCell key={index} col={col} id={index.toString()}>
                    <Box sx={{ display: 'inline-flex', alignItems: 'center' }}>
                      <Typography sx={{ mr: 0.7 }} variant="subtitle2">
                        {colTitle(col)}
                      </Typography>

                      {colFilters.length ? <FilterIndicator /> : null}
                      {sort && (
                        <SortingIndicator
                          sort={sort.sort}
                          sx={{ padding: 0.5 }}
                          size="small"
                        />
                      )}

                      {isColumnSortable(col) && (
                        <SortingButton
                          //? prevents the cell to drag when clicking on the button
                          beforeOpen={(e) => e.stopPropagation()}
                          col={col}
                        />
                      )}
                    </Box>
                  </DraggableCell>
                );
              })}
            </SortableRow>
          </TableHead>
        </TableFiltersContextProvider>
        <TableBody>
          {filteredRows.map((row) => (
            <TableRow key={rowKey(row)}>
              {displayedColumns.map((col, colidx) => (
                <TableCell
                  aria-labelledby={
                    typeof col.title === 'string'
                      ? columnId(col.title)
                      : undefined
                  }
                  sx={{ ...cellStyle, ...(col.customSx || {}) }}
                  key={`${rowKey(row)}-${colidx}`}
                >
                  {renderCol(row, col)}
                </TableCell>
              ))}
            </TableRow>
          ))}
          {!filteredRows.length && !loading && (
            <TableRow>
              <TableCell colSpan={displayedColumns.length}>
                {noData || <NoData />}
              </TableCell>
            </TableRow>
          )}
          {loading &&
            range(0, 3).map((idx) => (
              <TableRow key={`skel-row-${idx}`}>
                {displayedColumns.map((col, idx) => (
                  <TableCell key={`ske-${idx}`}>
                    {col.skeletonProps && <Skeleton {...col.skeletonProps} />}
                  </TableCell>
                ))}
              </TableRow>
            ))}
        </TableBody>
        <TableFooter>
          <TableRow>
            {!!pagination && (
              <TablePagination
                count={pagination.total}
                page={pagination.page}
                onPageChange={handlePageChange}
                rowsPerPage={pagination.rowsPerPage}
                onRowsPerPageChange={handleRowsPerPageChange}
              />
            )}
          </TableRow>
        </TableFooter>
      </MuiTable>
    </Box>
  );
};

export type { Column, TableProps };

export default Table;

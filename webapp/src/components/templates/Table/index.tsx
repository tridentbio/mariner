import { useEffect, useMemo, useState } from 'react';
import {
  Table as MuiTable,
  TableRow,
  TableBody,
  TableCell,
  TableHead,
  Box,
  TableFooter,
  TablePagination,
  TablePaginationProps,
  Skeleton,
} from '@mui/material';

import { Column, State, TableProps } from './types';
import FilterIndicator from './FilterIndicator';
import SortingIndicator from './SortingIndicator';
import NoData from 'components/atoms/NoData';
import { range } from '@utils';
import Filters from 'components/organisms/Table/Filters';
import { colTitle, columnId, isColumnSortable } from './common';
import SortingButton from './SortingButton';
const isKeyOf = <O,>(
  key: string | number | symbol | null,
  obj: O
): key is keyof O => {
  if (!key) return false;
  //@ts-ignore
  return key in obj;
};

const Table = <R extends { [key: string]: any }>({
  columns,
  rows,
  rowKey,
  filterModel,
  filterLinkOperatorOptions,
  sortModel,
  pagination,
  onStateChange,
  noData,
  loading,
  rowAlign,
  rowCellStyle,
  extraTableStyle,
}: TableProps<R>) => {
  const [state, setState] = useState<State>({
    filterModel: filterModel || { items: [] },
    sortModel: sortModel || [],
    paginationModel: pagination,
  });

  const isColumnInFilters = (col: Column<any, any>) =>
    state.filterModel.items.some((item) => item.columnName === col.field);

  const filterableColumns = useMemo(
    () => columns.filter((col) => !!col.filterSchema),
    [columns]
  );

  const handlePageChange: TablePaginationProps['onPageChange'] = (
    _event,
    newPage
  ) => {
    if (!state.paginationModel) {
      return;
    }
    const newState: State = {
      ...state,
      paginationModel: { ...state.paginationModel, page: newPage },
    };
    setState(newState);
  };
  const handleRowsPerPageChange: TablePaginationProps['onRowsPerPageChange'] = (
    _event
  ) => {
    if (!state.paginationModel) {
      return;
    }
    const newRowsPerPage = parseInt(_event.target.value);
    const newState: State = {
      ...state,
      paginationModel: {
        ...(state.paginationModel || { page: 0 }),
        rowsPerPage: newRowsPerPage,
      },
    };

    setState(newState);
  };

  const getColumnSorting = (col: Column<any, any>) =>
    state.sortModel.find((item) => item.field === col.field);

  const renderCol = (row: any, { render, field }: Column<any, any>) => {
    if (isKeyOf(field, row)) {
      return render ? render(row, row[field]) : row[field];
    } else if (render) {
      return render(row, null);
    } else {
      throw new Error(`Should have valid field or render. Either "${field}"\n
      is not a key of row, or there is no render`);
    }
  };

  if (!rowCellStyle) rowCellStyle = {};
  if (rowAlign === 'center') {
    rowCellStyle = {
      textAlign: 'center',
      verticalAlign: 'middle',
    };
  }

  const [detailed, setDetailed] = useState(false);

  const stateDependency = JSON.stringify(state);
  useEffect(() => {
    if (onStateChange) {
      onStateChange(state);
    }
  }, [stateDependency]);

  return (
    <div
      onMouseOver={() => setDetailed(true)}
      onMouseLeave={() => setDetailed(false)}
    >
      <MuiTable
        sx={{
          border: '1px solid rgb(224, 224, 224)',
          mb: 6,
          ...extraTableStyle,
        }}
      >
        <TableHead>
          {!!filterableColumns.length && (
            <TableRow>
              <TableCell sx={{ padding: '1px' }} colSpan={24}>
                <Filters
                  sortItems={state.sortModel}
                  filterItems={state.filterModel.items || []}
                  filterLinkOperatorOptions={filterLinkOperatorOptions || []}
                  detailed={detailed}
                  columns={columns}
                  filterableColumns={filterableColumns}
                  setState={setState}
                />
              </TableCell>
            </TableRow>
          )}
          <TableRow>
            {columns.map((col, index) => {
              const columnSort = getColumnSorting(col);
              // ! Big problems with filters and sort on table, the current implementation generate multiples popovers controlled by the same state and pointing to the same anchor, so they overlap each other being possible to access only the last popover, need to refactor table component
              return (
                <TableCell key={index} sx={col.customSx || {}}>
                  <Box sx={{ display: 'inline-flex', alignItems: 'center' }}>
                    {colTitle(col)}
                    {isColumnInFilters(col) && (
                      <FilterIndicator active={true} />
                    )}
                    {columnSort && <SortingIndicator sort={columnSort.sort} />}

                    {isColumnSortable(col) && (
                      <SortingButton
                        col={col}
                        sortState={state.sortModel}
                        setState={setState}
                      />
                    )}
                  </Box>
                </TableCell>
              );
            })}
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={rowKey(row)}>
              {columns.map((col, colidx) => (
                <TableCell
                  aria-labelledby={
                    typeof col.title === 'string'
                      ? columnId(col.title)
                      : undefined
                  }
                  sx={{ ...rowCellStyle, ...(col.customSx || {}) }}
                  key={`${rowKey(row)}-${colidx}`}
                >
                  {renderCol(row, col)}
                </TableCell>
              ))}
            </TableRow>
          ))}
          {!rows.length && !loading && (
            <TableRow>
              <TableCell colSpan={columns.length}>
                {noData || <NoData />}
              </TableCell>
            </TableRow>
          )}
          {loading &&
            range(0, 3).map((idx) => (
              <TableRow key={`skel-row-${idx}`}>
                {columns.map((col, idx) => (
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
    </div>
  );
};

export type { Column };

export default Table;

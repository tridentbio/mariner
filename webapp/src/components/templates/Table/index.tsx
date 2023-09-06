import {
  Box,
  Table as MuiTable,
  Skeleton,
  TableBody,
  TableCell,
  TableFooter,
  TableHead,
  TablePagination,
  TablePaginationProps,
  TableRow,
} from '@mui/material';
import { useEffect, useMemo, useState } from 'react';

import { useAppSelector } from '@app/hooks';
import { DraggableCell } from '@components/organisms/Table/DraggableCell';
import { SortableRow } from '@components/organisms/Table/SortableRow';
import { setPreference } from '@features/users/usersSlice';
import { useAppDispatch } from '@hooks';
import NoData from 'components/atoms/NoData';
import Filters, { FilterProps } from 'components/organisms/Table/Filters';
import { range } from 'utils';
import FilterIndicator from './FilterIndicator';
import SortingButton from './SortingButton';
import SortingIndicator from './SortingIndicator';
import { colTitle, columnId, isColumnSortable } from './common';
import { Column, State, TableProps } from './types';

const isKeyOf = <O,>(
  key: string | number | symbol | null,
  obj: O extends object ? O : never
): key is keyof O => {
  if (!key) return false;
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
  usePreferences,
  tableId,
}: TableProps<R>) => {
  const preferences = useAppSelector((state) => state.users.preferences);
  const dispatch = useAppDispatch();

  const [allColumns, setAllColumns] = useState<Column<any, any>[]>(columns);

  const columnsSortMatcher = (a: Column<any, any>, b: Column<any, any>) => {
    if (a.position === undefined || b.position === undefined) return 0;

    if (a.position > b.position) return 1;
    if (a.position < b.position) return -1;
    return 0;
  };

  const displayedColumns = useMemo<Column<any, any>[]>(() => {
    return allColumns.sort(columnsSortMatcher).filter((col) => !col.hidden);
  }, [allColumns]);

  const filterableColumns = useMemo(
    () => displayedColumns.filter((col) => !!col.filterSchema),
    [displayedColumns]
  );

  const [state, setState] = useState<State>({
    filterModel: filterModel || { items: [] },
    sortModel: sortModel || [],
    paginationModel: pagination,
  });

  const isColumnInFilters = (col: Column<any, any>) =>
    state.filterModel.items.some((item) => item.columnName === col.field);

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

  useEffect(() => {
    if (!usePreferences || !tableId) return;

    preferences.tables && onLoadedPreferences();
  }, [preferences.tables]);

  const onLoadedPreferences = () => {
    if (preferences.tables && tableId && preferences.tables[tableId]) {
      const tablePreferences = preferences.tables;

      if (tablePreferences) {
        //? Apply hidden col attributes
        const updatedColumns: Column<any, any>[] = allColumns.map((col) => {
          const foundPreferenceColRefIdx = tablePreferences[
            tableId
          ]?.columns.findIndex((c) => c.field === col.field);

          col.hidden = foundPreferenceColRefIdx == -1;

          if (foundPreferenceColRefIdx !== -1) {
            col.position = foundPreferenceColRefIdx;
          }

          return col;
        });

        setAllColumns(updatedColumns);
      }
    }
  };

  const onDroppedColumn = (reordenedDisplayedCols: Column<any, any>[]) => {
    //? Reorder columns
    const updatedColumns = allColumns.map((col, index) => {
      const foundColIdx = reordenedDisplayedCols.findIndex(
        (displayedCol) => displayedCol.field === col.field
      );

      if (foundColIdx !== -1)
        return {
          ...reordenedDisplayedCols[foundColIdx],
          position: foundColIdx,
        };

      return { ...col, position: undefined };
    }, []);

    setAllColumns(updatedColumns);

    if (usePreferences && tableId) {
      dispatch(
        setPreference({
          path: `tables.${tableId}`,
          data: {
            columns: reordenedDisplayedCols.map((col) => ({ field: col.field })),
          },
        })
      );
    }
  };

  const handleSelectedColumns: FilterProps['onSelectedColumns'] = (
    selectedColumnsIdList
  ) => {
    setAllColumns((prev) => {
      return prev.map((col) => {
        col.hidden = !selectedColumnsIdList.includes(col.field as string);

        return col;
      });
    });

    if (preferences?.tables && tableId && preferences.tables[tableId]) {
      dispatch(
        setPreference({
          path: `tables.${tableId}`,
          data: {
            columns: allColumns
              .filter((col) => selectedColumnsIdList.includes(col.field))
              .sort(columnsSortMatcher)
              .map((col) => ({ field: col.field })),
          },
        })
      );
    }
  };

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
          {(!!filterableColumns.length || !!state.sortModel) && (
            <TableRow>
              <TableCell sx={{ padding: 1 }} colSpan={24}>
                <Filters
                  sortItems={state.sortModel}
                  filterItems={state.filterModel.items || []}
                  filterLinkOperatorOptions={filterLinkOperatorOptions || []}
                  detailed={detailed}
                  columns={allColumns}
                  filterableColumns={filterableColumns}
                  onSelectedColumns={handleSelectedColumns}
                  setState={setState}
                />
              </TableCell>
            </TableRow>
          )}
          <SortableRow columns={displayedColumns} onDropped={onDroppedColumn}>
            {displayedColumns.map((col, index) => {
              const columnSort = getColumnSorting(col);

              // ! Big problems with filters and sort on table, the current implementation generate multiples popovers controlled by the same state and pointing to the same anchor, so they overlap each other being possible to access only the last popover, need to refactor table component
              return (
                <DraggableCell key={index} col={col} id={index.toString()}>
                  <Box sx={{ display: 'inline-flex', alignItems: 'center' }}>
                    {colTitle(col)}
                    {isColumnInFilters(col) && (
                      <FilterIndicator active={true} />
                    )}
                    {columnSort && <SortingIndicator sort={columnSort.sort} />}

                    {isColumnSortable(col) && (
                      <SortingButton
                        //? prevents the cell to drag when clicking on the button
                        beforeOpen={(e) => e.stopPropagation()}
                        col={col}
                        sortState={state.sortModel}
                        setState={setState}
                      />
                    )}
                  </Box>
                </DraggableCell>
              );
            })}
          </SortableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={rowKey(row)}>
              {displayedColumns.map((col, colidx) => (
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
    </div>
  );
};

export type { Column, TableProps };

export default Table;

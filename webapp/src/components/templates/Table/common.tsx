import { Typography } from '@mui/material';
import { ReactNode } from 'react';
import { title } from 'utils';
import { Column, FilterModel } from './types';

export const columnId = (title: string | number) => {
  return title.toString().toLowerCase().split(' ').join('-');
};

export const colTitle = (column: Column<any, any>, icon: ReactNode = null) => {
  if (typeof column.title === 'string')
    return (
      <>
        {icon}
        <Typography
          id={columnId(column.title)}
          whiteSpace="nowrap"
          variant="body2"
          sx={column.bold ? { fontWeight: 'bold' } : {}}
        >
          {title(column.title)}{' '}
        </Typography>
      </>
    );
  else if (column.title)
    return (
      <>
        {icon} {title(column.title as string)}
      </>
    );
  else
    return (
      <>
        {icon}
        <Typography id={columnId(column.title || '')} whiteSpace={'nowrap'}>
          {title(column.field)}
        </Typography>
      </>
    );
};

export const isColumnFilterable = (col: Column<any, any>) => !!col.filterSchema;
export const isColumnInFilters = (
  col: Column<any, any>,
  filterModel: FilterModel
) => filterModel.items.some((item) => item.columnName === col.field);
export const getFilterableColumns = (columns: Column<any, any>[]) =>
  columns.filter((col) => isColumnFilterable(col));

export const isColumnSortable = (col: Column<any, any>) => col.sortable;

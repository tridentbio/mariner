import { SkeletonProps } from '@mui/material';
import { CSSProperties } from '@mui/styled-engine';
import { SxProps, SystemStyleObject } from '@mui/system';
import { ReactNode } from 'react';

export type Column<
  R,
  K extends keyof R | null = null,
  O extends { key: string } = any
> = {
  render?: (row: R, value: any) => ReactNode;
  field?: K;
  name: string;
  title?: ReactNode;
  skeletonProps?: SkeletonProps;
  customSx?: SxProps;
  bold?: boolean;
  filterSchema?: {
    byValue?: boolean;
    byLessThan?: boolean;
    byGreaterThan?: boolean;
    byContains?: {
      options: O[];
      optionKey: (opt: O) => string | number;
      getLabel: (opt: O) => string;
    };
    // byContaining?: boolean
  };
  /**
   * Defines style and filtering inputs for the column
   * Defaults to text
   */
  type?: 'date' | 'number' | 'text';
  sortable?: boolean;
};

export type OperatorValue = 'eq' | 'lt' | 'gt' | 'ct' | 'inc';

export type FilterItem = {
  columnName: string;
  id?: number;
  operatorValue: OperatorValue;
  value: any;
};
export type FilterModel = {
  items: FilterItem[];
  linkOperator?: 'and' | 'or';
};

export type SortModel = { field: string; sort: 'asc' | 'desc' };

export type PaginationModel = {
  page: number;
  rowsPerPage: number;
  total: number;
};

export type State = {
  paginationModel?: PaginationModel;
  sortModel: SortModel[];
  filterModel: FilterModel;
};

export interface TableProps<R extends { [key: string]: any }> {
  filterModel?: FilterModel;
  filterLinkOperatorOptions?: ('and' | 'or')[];
  sortingMode?: 'client' | 'server';
  sortModel?: SortModel[];
  columns: Column<R, keyof R | null>[];
  rows: R[];
  rowKey: (row: R) => string | number;
  pagination?: PaginationModel;
  onStateChange?: (state: State) => void;
  noData?: ReactNode;
  loading?: boolean;
  rowAlign?: 'center' | 'start';
  rowCellStyle?: SystemStyleObject;
  extraTableStyle?: CSSProperties;
}

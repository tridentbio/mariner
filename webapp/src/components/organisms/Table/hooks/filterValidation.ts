import { TableProps } from '@components/templates/Table';
import { NonUndefined } from '@utils';
import { FilterModel } from '../../../templates/Table/types';
import {
  ColumnFilterManager,
  ContainsFilterStrategy,
  EqualsFilterStrategy,
  FilterStrategy,
  IncludesFilterStrategy,
} from './filterStrategies';

export const filterRows = <R extends { [key: string]: any }>(
  filterModel: FilterModel,
  columns: TableProps<R>['columns'],
  rows: TableProps<R>['rows'],
  dependencies: TableProps<R>['dependencies'] = {}
) => {
  let filteredRows = rows;

  columns.forEach((column) => {
    const colFilterParams = filterModel.items.find(
      (item) => item.columnName === column.field
    );

    if (!colFilterParams) return;

    const filterManager = new ColumnFilterManager();

    const filterTypes = column.filterSchema
      ? (Object.keys(column.filterSchema) as
          | (keyof NonUndefined<(typeof column)['filterSchema']>)[]
          | null)
      : null;

    filterTypes?.forEach((filterType) => {
      const filterSchema = column.filterSchema?.[filterType];

      if (filterSchema) {
        const strategy: FilterStrategy<unknown> | null = (() => {
          switch (filterType) {
            case 'byContains':
              return new ContainsFilterStrategy(colFilterParams.value);
            case 'byValue':
              return new EqualsFilterStrategy(colFilterParams.value);
            case 'byIncludes':
              return new IncludesFilterStrategy(colFilterParams.value);
            default:
              return null;
          }
        })();

        if (strategy) filterManager.addStrategy(strategy);
      }

      filteredRows = filteredRows.filter((row) => {
        const result = filterManager.validate(
          column.valueGetter
            ? column.valueGetter(row, dependencies)
            : row[column.field as keyof R]
        );

        return result;
      });
    });
  });

  return filteredRows;
};

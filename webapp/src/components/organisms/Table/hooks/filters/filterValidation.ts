import { TableProps } from '@components/templates/Table';
import { FilterModel } from '../../../../templates/Table/types';
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

  type ColumnFilterManagerDictRef = {
    colRef: TableProps<R>['columns'][number];
    filterManager: ColumnFilterManager<unknown>;
  };

  const columnsFilterManagerRef = columns.reduce<ColumnFilterManagerDictRef[]>(
    (acc, column) => {
      const colFilterParams = filterModel.items.filter(
        (item) => item.columnName === column.field
      );

      if (!colFilterParams.length) return acc;

      const filterManager = new ColumnFilterManager(filterModel.linkOperator);

      colFilterParams.forEach((filterItem) => {
        const strategy: FilterStrategy<unknown> | null = (() => {
          switch (filterItem.operatorValue) {
            case 'ct':
              return new ContainsFilterStrategy(filterItem.value);
            case 'eq':
              return new EqualsFilterStrategy(filterItem.value);
            case 'inc':
              return new IncludesFilterStrategy(filterItem.value);
            default:
              return null;
          }
        })();

        if (strategy) filterManager.addStrategy(strategy);
      });

      acc.push({ colRef: column, filterManager });

      return acc;
    },
    []
  );

  const defaultValidState = filterModel.linkOperator == 'and' ? true : false;

  filteredRows = filteredRows.filter((row) => {
    if (!columnsFilterManagerRef.length) return true;

    const validRow = columnsFilterManagerRef.reduce<boolean>(
      (acc, { colRef, filterManager }) => {
        const filterResult = filterManager.validate(
          colRef.valueGetter
            ? colRef.valueGetter(row, dependencies)
            : row[colRef.field as keyof R]
        );

        return filterModel.linkOperator == 'and'
          ? acc && filterResult
          : acc || filterResult;
      },
      defaultValidState
    );

    return validRow;
  });

  return filteredRows;
};

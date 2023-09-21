import { Experiment, TrainingStage } from '@app/types/domain/experiments';
import {
  Column,
  FilterItem,
  FilterModel,
} from '@components/templates/Table/types';
import { expect, test } from '@jest/globals';
import { rows as allRows } from '../../../../../../tests/fixtures/table/rowsMock';
import { filterRows } from './filterValidation';

describe('Table filter validation', () => {
  const columns = [
    {
      title: 'Experiment Name',
      name: 'Experiment Name',
      field: 'experimentName',
      filterSchema: {
        byIncludes: true,
      },
    },
    {
      field: 'stage',
      title: 'Stage',
      name: 'Stage',
      filterSchema: {
        byContains: {
          options: [
            { label: 'Trained', key: 'SUCCESS' },
            { label: 'Training', key: 'RUNNING' },
            { label: 'Failed', key: 'ERROR' },
            { label: 'Not started', key: 'NOT RUNNING' },
          ],
          optionKey: (option) => option.key,
          getLabel: (option) => option.label,
        },
      },
    },
    {
      field: 'trainMetrics' as const,
      name: 'Train Loss',
      title: 'Train Loss',
    },
    {
      field: 'trainMetrics' as const,
      name: 'Validation Loss',
      title: 'Validation Loss',
    },
    {
      name: 'Learning Rate',
      field: 'hyperparams' as const,
      title: 'LR',
    },
    {
      name: 'Epochs',
      field: 'epochs' as const,
      title: 'Epochs',
      filterSchema: {
        byContains: {
          options: [
            { label: 'ITEM1', key: 9 },
            { label: 'ITEM2', key: 4 },
          ],
          optionKey: (option) => option.key,
          getLabel: (option) => option.label,
        },
      },
    },
    {
      name: 'Created At',
      field: 'createdAt' as const,
      title: 'Created At',
    },
  ] as Column<Experiment, keyof Experiment>[];

  const rows = allRows as Experiment[];

  //TODO: Declare the rest of the operators when they are implemented in the UI
  const filterItems: { [operator in 'inc' | 'ct']: FilterItem } = {
    inc: {
      columnName: 'experimentName',
      operatorValue: 'inc',
      value: '2',
    },
    ct: {
      columnName: 'stage',
      operatorValue: 'ct',
      value: ['SUCCESS', 'ERROR'],
    },
  };

  const mountFilterModel = (
    linkOperator: FilterModel['linkOperator'],
    items: FilterItem[]
  ) =>
    ({
      linkOperator,
      items,
    } as FilterModel);

  it('should not filter rows if filterModel is empty', () => {
    const filterModel = mountFilterModel('and', []);

    const filteredRows = filterRows(filterModel, columns, rows);

    expect(filteredRows.length).toBe(rows.length);
  });

  describe('Include operator', () => {
    test('AND', () => {
      const filterModel = mountFilterModel('and', [filterItems.inc]);
      const experimentFilter = filterModel.items[0].value;

      const filteredRows = filterRows(filterModel, columns, rows);
      const unfilteredRows = filteredRows.filter(
        (row) => !row.experimentName?.includes(experimentFilter)
      );

      expect(unfilteredRows.length).toBe(0);
    });

    test('OR', () => {
      const filterModel = mountFilterModel('or', [
        filterItems.inc,
        { ...filterItems.inc, value: 'Test' },
      ]);
      const experimentFilter = filterModel.items[0].value;
      const experimentFilter2 = filterModel.items[1].value;

      const filteredRows = filterRows(filterModel, columns, rows);
      const unfilteredRows = filteredRows.filter(
        (row) =>
          !row.experimentName?.includes(experimentFilter) &&
          !row.experimentName?.includes(experimentFilter2)
      );

      expect(unfilteredRows.length).toBe(0);
    });
  });

  describe('Contains operator', () => {
    test('AND', () => {
      const filterModel = mountFilterModel('and', [filterItems.ct]);
      const stages: TrainingStage[] = filterModel.items[0]
        .value as TrainingStage[];

      const rowsWithValidStage = rows.filter((row) =>
        stages.includes(row.stage)
      );

      const filteredRows = filterRows(
        filterModel,
        columns,
        rows as Experiment[]
      );
      const unfilteredRows = filteredRows.filter(
        (row) => !stages.includes(row.stage)
      );

      expect(unfilteredRows.length).toBe(0);
      expect(filteredRows.length).toBe(rowsWithValidStage.length);
    });

    test('OR', () => {
      const filterModel = mountFilterModel('or', [
        { ...filterItems.ct, value: ['SUCCESS'] },
        { ...filterItems.ct, columnName: 'epochs', value: [4] },
      ]);

      const trainingStages: TrainingStage[] = filterModel.items[0]
        .value as TrainingStage[];
      const epochs: number[] = filterModel.items[1].value as number[];

      const rowsWithValidStage = rows.filter((row) =>
        trainingStages.includes(row.stage)
      );

      const orFilteredRows = filterRows(
        filterModel,
        columns,
        rows as Experiment[]
      );
      const orUnfilteredRows = orFilteredRows.filter(
        (row) =>
          !trainingStages.includes(row.stage) &&
          row.epochs &&
          !epochs.includes(row.epochs)
      );

      expect(orUnfilteredRows.length).toBe(0);
      expect(orFilteredRows.length).toBe(rowsWithValidStage.length);
    });
  });

  describe('Multiple filters', () => {
    test('AND', () => {
      const filterModel = mountFilterModel('and', [
        filterItems.inc,
        filterItems.ct,
      ]);
      const experimentFilter = filterModel.items[0].value;
      const stages: TrainingStage[] = filterModel.items[1]
        .value as TrainingStage[];

      const filteredRows = filterRows(filterModel, columns, rows);
      const unfilteredRows = filteredRows.filter(
        (row) =>
          !row.experimentName?.includes(experimentFilter) ||
          !stages.includes(row.stage)
      );

      expect(unfilteredRows.length).toBe(0);
    });

    test('OR', () => {
      const filterModel = mountFilterModel('or', [
        filterItems.inc,
        filterItems.ct,
        { ...filterItems.inc, value: '1' },
      ]);
      const experimentFilters = [
        filterModel.items[0].value,
        filterModel.items[2].value,
      ];
      const stages: TrainingStage[] = filterModel.items[1]
        .value as TrainingStage[];

      const filteredRows = filterRows(filterModel, columns, rows);
      const unfilteredRows = filteredRows.filter(
        (row) =>
          !experimentFilters.includes(row.experimentName || '') &&
          !stages.includes(row.stage)
      );

      expect(unfilteredRows.length).toBe(0);
    });
  });
});

import { Chip } from '@mui/material';
import React from 'react';
import {
  Column,
  FilterItem,
  OperatorValue,
} from 'components/templates/Table/types';

type ChipFilterContainProps = {
  onDelete: () => void;
  filterItem: FilterItem;
  column: Column<any, any>;
  generateOperationTitle: (op: OperatorValue) => string;
};

const ChipFilterContain: React.FC<ChipFilterContainProps> = ({
  onDelete,
  filterItem,
  column,
  generateOperationTitle,
}) => {
  return (
    <Chip
      data-testid={`chip-filter-${column.name}`}
      onDelete={onDelete}
      sx={{ mr: 1, py: 1, fontSize: 14 }}
      label={`${column?.name || filterItem.columnName} ${generateOperationTitle(
        filterItem.operatorValue
      )} ${filterItem.value
        .map((itemValueKey: { key: string }[]) => {
          if (!column) {
            return itemValueKey;
          }
          if (column.filterSchema?.byContains?.options) {
            const itemValue = column.filterSchema?.byContains.options.find(
              (opt) => opt.key === itemValueKey
            );
            return itemValue
              ? column.filterSchema.byContains.getLabel(itemValue)
              : itemValueKey;
          }
        })
        .map((str: string) => `"${str}"`)
        .join(', ')}`}
    />
  );
};

export default ChipFilterContain;

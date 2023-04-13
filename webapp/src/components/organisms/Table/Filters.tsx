import {
  CalendarViewDayOutlined,
  NumbersOutlined,
  ShortTextOutlined,
} from '@mui/icons-material';
import {
  Button,
  Chip,
  MenuItem,
  MenuList,
  Popover,
  Link,
  TextField,
} from '@mui/material';
import { Box } from '@mui/system';
import React, { useState } from 'react';
import { title } from '@utils';
import ChipFilterContain from './ChipFilterContain';
import ColumnFiltersInput from 'components/templates/Table/ColumnFiltersInput';
import { colTitle } from 'components/templates/Table/common';
import {
  Column,
  FilterItem,
  OperatorValue,
  SortModel,
  State,
} from 'components/templates/Table/types';
import { usePopoverState } from '@hooks/usePopoverState';

export interface FilterProps {
  filterLinkOperatorOptions: ('and' | 'or')[];
  filterItems: FilterItem[];
  columns: Column<any, any>[];
  detailed?: boolean;
  sortItems: SortModel[];
  filterableColumns: Column<any, any>[];
  setState: React.Dispatch<React.SetStateAction<State>>;
}
const Filters = ({
  filterLinkOperatorOptions,
  columns,
  filterItems,
  detailed,
  sortItems,
  filterableColumns,
  setState,
}: FilterProps) => {
  const addFilterPopover = usePopoverState();
  const columnFilterPopover = usePopoverState();
  const [selectedColumn, setSelectedColumn] = useState<Column<any, any>>();
  const [linkOperator, setLinkOperator] = useState<'and' | 'or'>('and');

  const colIcon = ({ type }: Column<any, any>) => {
    if (!type) return <ShortTextOutlined />;
    const colIconMap = {
      date: <CalendarViewDayOutlined />,
      number: <NumbersOutlined />,
      text: <ShortTextOutlined />,
    };
    return colIconMap[type] || <ShortTextOutlined />;
  };

  const singleFilterLinkOption =
    !filterLinkOperatorOptions || filterLinkOperatorOptions.length === 1;

  const operationTitle = (op: OperatorValue) => {
    const operationMap = {
      eq: 'equals',
      lt: 'less than',
      gt: 'greater than',
      ct: 'is one of',
      inc: 'includes',
    };
    return operationMap[op as keyof typeof operationMap];
  };

  const onOpenColumnFilterMenu = (
    event: React.MouseEvent<any>,
    columnField: Column<any, any>['field']
  ) => {
    const column = columns.find((item) => item.field === columnField);
    if (!column) return;
    setSelectedColumn(column);
    columnFilterPopover.setAnchorEl(event.currentTarget);
  };

  const onAddFilter = (
    columnField: string,
    operatorValue: OperatorValue,
    value: any
  ) => {
    setState((prev) => ({
      ...prev,
      filterModel: {
        items: [
          ...prev.filterModel.items,
          {
            columnName: columnField,
            operatorValue,
            value,
          },
        ],
        linkOperator: 'and',
      },
    }));
  };

  const onFilterLinkChange = (newLink: 'and' | 'or') => {
    setState((prev) => ({
      ...prev,
      filterModel: { ...prev.filterModel, linkOperator: newLink },
    }));
  };

  return (
    <>
      <Popover
        anchorOrigin={{
          vertical: 'center',
          horizontal: 'right',
        }}
        open={addFilterPopover.open}
        anchorEl={addFilterPopover.anchorEl}
        onClose={addFilterPopover.handleClose}
      >
        <Box sx={{ padding: 1 }}>
          <TextField
            select
            variant="standard"
            sx={{ width: '100%' }}
            value={linkOperator}
            disabled={singleFilterLinkOption}
            onChange={(event) =>
              onFilterLinkChange(event.target.value as 'and' | 'or')
            }
          >
            {(filterLinkOperatorOptions || ['and']).map((op) => (
              <MenuItem key={op} value={op}>
                {title(op)}
              </MenuItem>
            ))}
          </TextField>
          <MenuList>
            {filterableColumns.map((col) => (
              <MenuItem
                onClick={(event) => onOpenColumnFilterMenu(event, col.field)}
                key={col.field as string}
              >
                <Box sx={{ display: 'inline-flex' }}>
                  {colTitle(col, colIcon(col))}
                </Box>
              </MenuItem>
            ))}
          </MenuList>
        </Box>
      </Popover>
      <Popover
        anchorOrigin={{ vertical: 'center', horizontal: 'right' }}
        open={columnFilterPopover.open}
        anchorEl={columnFilterPopover.anchorEl}
        onClose={columnFilterPopover.handleClose}
      >
        {selectedColumn && (
          <ColumnFiltersInput col={selectedColumn} onAddFilter={onAddFilter} />
        )}
      </Popover>

      <Box sx={{ width: '100%' }}>
        {filterItems?.map((item, index) => {
          const column = columns.find((col) => col?.field === item.columnName);
          if (!column) return;
          return item.operatorValue === 'ct' ? (
            <ChipFilterContain
              onDelete={() =>
                setState((prev) => ({
                  ...prev,
                  filterModel: {
                    items: filterItems.filter(
                      (_, itemIndex) => itemIndex !== index
                    ),
                  },
                }))
              }
              key={
                item.columnName +
                item.operatorValue +
                String(item.value) +
                String(item.id)
              }
              filterItem={item}
              column={column}
              generateOperationTitle={operationTitle}
            />
          ) : (
            <Chip
              onDelete={() =>
                setState((prev) => ({
                  ...prev,
                  filterModel: {
                    items: filterItems.filter(
                      (_, itemIndex) => itemIndex !== index
                    ),
                  },
                }))
              }
              sx={{ mb: 1, mr: 1 }}
              key={
                item.columnName +
                item.operatorValue +
                String(item.value) +
                String(item.id)
              }
              label={`"${item.columnName}" ${operationTitle(
                item.operatorValue
              )} ${item.value}`}
            />
          );
        })}
        <Button
          onClick={addFilterPopover.handleClickOpenPopover}
          sx={{
            padding: 1,
            textTransform: 'none',
            color: detailed ? 'primary.main' : 'rgba(0,0,0,0)',
            transition: 'color 0.5s',
          }}
          variant="text"
        >
          Add Filter
        </Button>

        {filterItems?.length > 0 && (
          <Button
            sx={{
              padding: 1,
              textTransform: 'none',
              color: detailed ? 'primary.main' : 'rgba(0,0,0,0)',
              transition: 'color 0.5s',
            }}
            variant="text"
            onClick={() =>
              setState((prev) => ({ ...prev, filterModel: { items: [] } }))
            }
          >
            Clear all filters
          </Button>
        )}
      </Box>
      <Box sx={{ width: '100%' }}>
        {sortItems.map((item: SortModel, index) => {
          const column = columns.find((col) => col.field === item.field);
          if (!column) return null;
          return (
            <Chip
              onDelete={(_e) =>
                setState((prev) => ({
                  ...prev,
                  sortModel: prev.sortModel.filter(
                    (sorted) => sorted.field !== item.field
                  ),
                }))
              }
              sx={{ mb: 1, mr: 1 }}
              key={column.name + String(item.sort)}
              label={`${column.name} ${item.sort.toUpperCase()}`}
            />
          );
        })}

        {sortItems.length > 0 && (
          <Link
            sx={{ cursor: 'pointer' }}
            onClick={() => setState((prev) => ({ ...prev, sortModel: [] }))}
          >
            Clear all sortings
          </Link>
        )}
      </Box>
    </>
  );
};
export default Filters;

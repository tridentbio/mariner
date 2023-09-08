import {
  Add,
  ExpandMore,
  FormatListBulleted,
  RemoveCircle,
} from '@mui/icons-material';
import { Button, Chip } from '@mui/material';
import { Box } from '@mui/system';
import {
  Column,
  FilterItem,
  OperatorValue,
  SortModel,
  State,
} from 'components/templates/Table/types';
import { usePopoverState } from 'hooks/usePopoverState';
import React, { useMemo, useRef } from 'react';
import ChipFilterContain from './ChipFilterContain';
import { ColumnPicker } from './ColumnPicker';
import { OperatorsFilterMenu } from './OperatorsFilterMenu';

export interface FilterProps {
  filterLinkOperatorOptions?: ('and' | 'or')[];
  filterItems: FilterItem[];
  columns: Column<any, any>[];
  sortItems: SortModel[];
  filterableColumns: Column<any, any>[];
  onSelectedColumns?: (columnsIdList: string[]) => void;
  setState: React.Dispatch<React.SetStateAction<State>>;
}

/* const StyledHeaderOptionButton = styled(Button)(({ theme }) => ({
  bborderRadius: 2,
  paddingX: 3,
})); */

const Filters = ({
  filterLinkOperatorOptions,
  columns,
  filterItems,
  sortItems,
  filterableColumns,
  setState,
  onSelectedColumns,
}: FilterProps) => {
  const addFilterPopover = usePopoverState();
  const columnPickerPopover = usePopoverState();

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

  const columnsTreeView = useMemo<TreeNode[]>(() => {
    return [
      {
        id: 'menu',
        name: 'Test',
        children: [
          {
            id: 'menu-2',
            name: 'Test 2',
            children: columns
              .filter((col) => !col.fixed)
              .map((column) => ({
                id: column.name as string,
                name: column.name,
                parent: 'menu',
              })),
          },
        ],
      },
    ];
  }, [columns]);

  const defaultSelectedColumns = columns
    .filter((col) => !col.hidden)
    .map((col) => col.name as string);

  const displayedColumns = useRef<Column<any, any>[]>(columns);

  return (
    <>
      <ColumnPicker
        popoverProps={{
          anchorEl: columnPickerPopover.anchorEl,
          anchorOrigin: {
            vertical: 'bottom',
            horizontal: 'left',
          },
          onClose: columnPickerPopover.handleClose,
        }}
        open={columnPickerPopover.open}
        treeView={columnsTreeView}
        height={480}
        onChange={(selectedColumns) => {
          displayedColumns.current = columns.filter((col) =>
            selectedColumns.includes(col.name)
          );

          onSelectedColumns && onSelectedColumns(selectedColumns);
        }}
        defaultSelectedColumns={defaultSelectedColumns}
      />

      <Box sx={{ width: '100%' }}>
        {filterableColumns?.length ? (
          <>
            <OperatorsFilterMenu
              open={addFilterPopover.open}
              anchorEl={addFilterPopover.anchorEl}
              onClose={addFilterPopover.handleClose}
              columns={columns}
              filterLinkOperatorOptions={filterLinkOperatorOptions}
              filterableColumns={filterableColumns}
              setState={setState}
            />
            <Button
              startIcon={<Add />}
              onClick={addFilterPopover.handleClickOpenPopover}
              sx={{
                borderRadius: 2,
                paddingX: 3,
              }}
              variant="text"
            >
              Add Filter
            </Button>
          </>
        ) : null}

        <Button
          startIcon={<FormatListBulleted />}
          endIcon={<ExpandMore />}
          sx={{
            borderRadius: 2,
            paddingX: 3,
          }}
          color="primary"
          onClick={columnPickerPopover.handleClickOpenPopover}
        >
          Columns
        </Button>

        {filterItems?.length > 0 && (
          <Button
            startIcon={<RemoveCircle />}
            size="small"
            sx={{
              borderRadius: 2,
              paddingX: 3,
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

      {filterItems?.length > 0 ? (
        <Box pt={0.8}>
          {filterItems.map((item, index) => {
            const column = displayedColumns.current.find(
              (col) => col?.field === item.columnName
            );
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
                key={index}
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
                sx={{ mr: 1 }}
                key={index}
                label={`"${item.columnName}" ${operationTitle(
                  item.operatorValue
                )} ${item.value}`}
              />
            );
          })}
        </Box>
      ) : null}

      <Box
        sx={{
          width: '100%',
          paddingTop: sortItems.length ? 1 : 0,
        }}
      >
        {sortItems.map((item: SortModel, index) => {
          const column = displayedColumns.current.find(
            (col) => col.field === item.field
          );
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
              sx={{ mr: 1, py: 1, fontSize: 14 }}
              key={column.name + String(item.sort)}
              label={`${column.name} ${item.sort.toUpperCase()}`}
            />
          );
        })}

        {sortItems.length > 0 && (
          <Chip
            sx={{ fontSize: 15 }}
            label="Clear all sortings"
            icon={<RemoveCircle fontSize="small" />}
            onClick={() => setState((prev) => ({ ...prev, sortModel: [] }))}
            color="primary"
            variant="outlined"
          />
        )}
      </Box>
    </>
  );
};
export default Filters;

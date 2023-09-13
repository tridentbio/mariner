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
  OperatorValue,
  SortModel,
} from 'components/templates/Table/types';
import { usePopoverState } from 'hooks/usePopoverState';
import { useContext, useRef } from 'react';
import ChipFilterContain from './ChipFilterContain';
import { ColumnPicker } from './ColumnPicker';
import { OperatorsFilterMenu } from './OperatorsFilterMenu';
import { TableFilterContext } from './hooks/useTableFilters';

export interface FilterProps {
  filterLinkOperatorOptions?: ('and' | 'or')[];
  columns: Column<any, any>[];
  onSelectedColumns?: (columnsIdList: string[]) => void;
  treeView?: TreeNode[];
}

const Filters = ({
  filterLinkOperatorOptions,
  columns,
  onSelectedColumns,
  treeView,
}: FilterProps) => {
  const addFilterPopover = usePopoverState();
  const columnPickerPopover = usePopoverState();

  const {
    filters: { filterModel, sortModel },
    setFilters,
    filterableColumns,
  } = useContext(TableFilterContext);

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

  const displayedColumns = useRef<Column<any, any>[]>(columns);

  const isPickableColumn = (column: Column<any, any>) =>
    !column.hidden && !column.fixed;

  const defaultTreeView = useRef(
    columns.filter(isPickableColumn).map((column) => ({
      id: column.name as string,
      name: column.name,
    }))
  );

  const defaultSelectedColumns = columns
    .filter(isPickableColumn)
    .map((col) => col.name as string);

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
        treeView={treeView || defaultTreeView.current}
        height={480}
        onChange={(selectedColumns) => {
          displayedColumns.current = columns.filter((col) =>
            selectedColumns.includes(col.name)
          );

          onSelectedColumns && onSelectedColumns(selectedColumns);
        }}
        defaultSelectedColumns={defaultSelectedColumns}
      />

      <Box
        sx={{
          width: 'fit-content',
          position: 'sticky',
          left: 0,
        }}
      >
        <Box>
          {filterableColumns?.length ? (
            <>
              <OperatorsFilterMenu
                open={addFilterPopover.open}
                anchorEl={addFilterPopover.anchorEl}
                onClose={addFilterPopover.handleClose}
                columns={columns}
                filterLinkOperatorOptions={filterLinkOperatorOptions}
              />
              <Button
                data-testid="add-filter-btn"
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
            data-testid="column-picker-btn"
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

          {filterModel.items?.length > 0 && (
            <Button
              data-testid="clear-all-filters-btn"
              startIcon={<RemoveCircle />}
              size="small"
              sx={{
                borderRadius: 2,
                paddingX: 3,
              }}
              variant="text"
              onClick={() =>
                setFilters((prev) => ({ ...prev, filterModel: { items: [] } }))
              }
            >
              Clear all filters
            </Button>
          )}
        </Box>

        {filterModel.items?.length > 0 ? (
          <Box pt={0.8}>
            {filterModel.items.map((item, index) => {
              const column = displayedColumns.current.find(
                (col) => col?.field === item.columnName
              );
              if (!column) return;

              return item.operatorValue === 'ct' ? (
                <ChipFilterContain
                  onDelete={() =>
                    setFilters((prev) => ({
                      ...prev,
                      filterModel: {
                        items: filterModel.items.filter(
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
                  sx={{ mr: 1, py: 1, fontSize: 14 }}
                  data-testid={`chip-filter-${column.name}`}
                  onDelete={() =>
                    setFilters((prev) => ({
                      ...prev,
                      filterModel: {
                        items: filterModel.items.filter(
                          (_, itemIndex) => itemIndex !== index
                        ),
                      },
                    }))
                  }
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
            paddingTop: sortModel.length ? 1 : 0,
          }}
        >
          {sortModel.map((item: SortModel, index) => {
            const column = displayedColumns.current.find(
              (col) => col.field === item.field
            );
            if (!column) return null;
            return (
              <Chip
                data-testid={`chip-sort-${column.name}`}
                onDelete={(_e) =>
                  setFilters((prev) => ({
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

          {sortModel.length > 0 && (
            <Chip
              sx={{ fontSize: 15 }}
              label="Clear all sortings"
              icon={<RemoveCircle fontSize="small" />}
              onClick={() => setFilters((prev) => ({ ...prev, sortModel: [] }))}
              color="primary"
              variant="outlined"
            />
          )}
        </Box>
      </Box>
    </>
  );
};
export default Filters;

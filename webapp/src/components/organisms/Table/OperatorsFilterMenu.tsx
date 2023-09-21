import { Column } from '@components/templates/Table';
import ColumnFiltersInput from '@components/templates/Table/ColumnFiltersInput';
import { colTitle } from '@components/templates/Table/common';
import { FilterItem, OperatorValue } from '@components/templates/Table/types';
import { usePopoverState } from '@hooks';
import {
  CalendarViewDayOutlined,
  NumbersOutlined,
  ShortTextOutlined,
} from '@mui/icons-material';
import {
  Box,
  ClickAwayListener,
  Grow,
  MenuItem,
  MenuList,
  Popover,
  Popper,
  TextField,
  styled,
} from '@mui/material';
import { useContext, useMemo, useState } from 'react';
import { NonUndefined, title, uniqBy } from 'utils';
import { FilterProps } from './Filters';
import { TableStateContext } from './hooks/useTableState';

interface OperatorsFilterMenuProps
  extends Pick<FilterProps, 'filterLinkOperatorOptions' | 'columns'> {
  open: boolean;
  anchorEl: HTMLElement | null;
  onClose?: () => void;
}

const StyledMenuBox = styled(Box)(({ theme }) => ({
  '.MuiTypography-root, .MuiSelect-select, .MuiFormLabel-root, .MuiMenuItem-root':
    {
      fontSize: theme.typography.body2.fontSize,
    },
}));

type OperatorOptions = NonUndefined<FilterProps['filterLinkOperatorOptions']>;

export const OperatorsFilterMenu = ({
  open,
  anchorEl,
  onClose,
  filterLinkOperatorOptions,
  columns,
}: OperatorsFilterMenuProps) => {
  const {
    setFilters,
    filterableColumns,
    filters: { filterModel },
  } = useContext(TableStateContext);

  const operatorOptions = useMemo<OperatorOptions>(() => {
    if (!filterLinkOperatorOptions) return [filterModel.linkOperator];
    return filterLinkOperatorOptions;
  }, [filterLinkOperatorOptions]);

  const columnFilterPopper = usePopoverState();
  const [selectedColumn, setSelectedColumn] = useState<Column<any, any>>();

  //? ref being used as a state to trigger rerender and avoid the select input dropdown being placed incorrectly on DOM
  const [columnFilterBoxRef, setColumnFilterBoxRef] = useState<HTMLElement>();

  const colIcon = ({ type }: Column<any, any>) => {
    if (!type) return <ShortTextOutlined />;
    const colIconMap = {
      date: <CalendarViewDayOutlined />,
      number: <NumbersOutlined />,
      text: <ShortTextOutlined />,
    };
    return colIconMap[type] || <ShortTextOutlined />;
  };

  const onOpenColumnFilterMenu = (
    event: React.MouseEvent<any>,
    columnField: Column<any, any>['field']
  ) => {
    const column = columns.find((item) => item.field === columnField);
    if (!column) return;
    setSelectedColumn(column);

    if (!columnFilterPopper.open) {
      columnFilterPopper.setAnchorEl(event.currentTarget);
      setColumnFilterBoxRef(event.currentTarget);
    }
  };

  const onAddFilter = (
    columnField: string,
    operatorValue: OperatorValue,
    value: any
  ) => {
    setFilters((prev) => {
      if (operatorValue === 'ct') {
        let filterItem = prev.filterModel.items.find(
          (item) => item.columnName == columnField
        );

        if (filterItem?.value) {
          //? Add new elements and remove duplicated ones
          filterItem.value = uniqBy(
            (filterItem?.value as string[]).concat(value as string[]),
            (item) => item
          );

          return { ...prev };
        }
      }

      prev.filterModel.items.push({
        columnName: columnField,
        operatorValue,
        value,
      });

      return { ...prev };
    });

    columnFilterPopper.handleClose();
  };

  const onFilterLinkChange = (newLink: 'and' | 'or') => {
    setFilters((prev) => ({
      ...prev,
      filterModel: { ...prev.filterModel, linkOperator: newLink },
    }));
  };

  const handleClose = (event: Event | React.SyntheticEvent) => {
    if (
      columnFilterBoxRef &&
      columnFilterBoxRef.contains(event.target as HTMLElement)
    ) {
      return;
    }

    onClose && onClose();

    columnFilterPopper.handleClose();

    setColumnFilterBoxRef(undefined);
  };

  return (
    <Popover
      anchorOrigin={{
        vertical: 'center',
        horizontal: 'right',
      }}
      open={open}
      anchorEl={anchorEl}
      onClose={handleClose}
      PaperProps={{
        sx: { overflow: 'initial' },
      }}
    >
      <StyledMenuBox
        sx={{ padding: 2 }}
        ref={(e: HTMLElement | null) => {
          e && setColumnFilterBoxRef(e);
        }}
      >
        <TextField
          select
          variant="standard"
          sx={{ width: '100%' }}
          value={filterModel.linkOperator}
          disabled={operatorOptions.length === 1}
          onChange={(event) =>
            onFilterLinkChange(event.target.value as 'and' | 'or')
          }
          SelectProps={{
            MenuProps: {
              container: columnFilterBoxRef,
            },
          }}
        >
          {operatorOptions.map((op) => (
            <MenuItem key={op} value={op}>
              {title(op)}
            </MenuItem>
          ))}
        </TextField>
        {filterableColumns?.length ? (
          <MenuList sx={{ pb: 0 }}>
            {filterableColumns.map((col) => (
              <MenuItem
                data-testid={`add-filter-${col.name}-option`}
                onClick={(event) => onOpenColumnFilterMenu(event, col.field)}
                key={col.name as string}
              >
                <Box sx={{ display: 'inline-flex', alignItems: 'center' }}>
                  {colTitle(col, colIcon(col))}
                </Box>
              </MenuItem>
            ))}
          </MenuList>
        ) : null}
      </StyledMenuBox>

      {columnFilterBoxRef ? (
        <Popper
          open={columnFilterPopper.open}
          anchorEl={columnFilterBoxRef}
          container={columnFilterBoxRef}
          role={undefined}
          placement="right-end"
          transition
        >
          {({ TransitionProps, placement }) => (
            <Grow {...TransitionProps} style={{ transformOrigin: 'right' }}>
              <Box>
                <ClickAwayListener onClickAway={handleClose}>
                  <Box>
                    {selectedColumn && (
                      <ColumnFiltersInput
                        col={selectedColumn}
                        onAddFilter={onAddFilter}
                      />
                    )}
                  </Box>
                </ClickAwayListener>
              </Box>
            </Grow>
          )}
        </Popper>
      ) : null}
    </Popover>
  );
};

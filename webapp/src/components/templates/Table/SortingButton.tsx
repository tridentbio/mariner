import { TableFilterContext } from '@components/organisms/Table/hooks/useTableFilters';
import { ArrowDownward, ArrowUpward, MoreVert } from '@mui/icons-material';
import {
  Box,
  IconButton,
  IconButtonProps,
  MenuItem,
  MenuList,
  Popover,
  styled,
} from '@mui/material';
import React, { MouseEvent, useContext } from 'react';
import { Column } from './types';

interface SortingButtonProps extends Pick<IconButtonProps, 'sx' | 'size'> {
  col: Column<any, any>;
  beforeOpen?: (e: MouseEvent) => void;
}

const StyledMenuItem = styled(MenuItem)(({ theme }) => ({
  fontSize: '1rem',
}));

const SortingButton: React.FC<SortingButtonProps> = ({
  col,
  beforeOpen,
  sx,
  size = 'small',
}) => {
  const [anchorEl, setAnchorEl] = React.useState<HTMLButtonElement | null>(
    null
  );

  const {
    filters: { sortModel },
    setFilters,
  } = useContext(TableFilterContext);

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };
  const open = Boolean(anchorEl);

  const isSelectedSorting = (
    col: Column<any, any>,
    ordering: 'asc' | 'desc'
  ): boolean => {
    return sortModel.some(
      (item) => item.field === col.field && ordering === item.sort
    );
  };

  const handleSelectSort = (
    col: Column<any, any>,
    ordering: 'asc' | 'desc'
  ) => {
    let alreadySortedByThisField = false;
    const newSortState = sortModel.map((item) => {
      if (item.field === col.field) {
        alreadySortedByThisField = true;
        item.sort = ordering;
      }
      return item;
    });
    if (!alreadySortedByThisField) {
      newSortState.push({ field: col.field, sort: ordering });
    }
    setFilters((prev) => ({ ...prev, sortModel: newSortState }));
  };
  return (
    <Box data-testid={`sorting-button-${col.name}`} onMouseDown={beforeOpen}>
      <IconButton
        sx={{
          ...(sx || {}),
          color: 'gray',
          transition: 'color 0.5s',
          padding: 0.5,
        }}
        onClick={handleClick}
      >
        <MoreVert fontSize={size} />
      </IconButton>
      <Popover
        anchorOrigin={{
          vertical: 'center',
          horizontal: 'right',
        }}
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        sx={{
          fontSize: '0.8rem',
        }}
      >
        <MenuList>
          <StyledMenuItem
            data-testid={`sort-asc-${col.name}`}
            selected={isSelectedSorting(col, 'asc')}
            onClick={() => {
              handleSelectSort(col, 'asc');
              handleClose();
            }}
          >
            <ArrowUpward fontSize={size} sx={{ mr: 0.5 }} />
            Sort Asc
          </StyledMenuItem>
          <StyledMenuItem
            data-testid={`sort-desc-${col.name}`}
            selected={isSelectedSorting(col, 'desc')}
            onClick={() => {
              handleSelectSort(col, 'desc');
              handleClose();
            }}
          >
            <ArrowDownward fontSize={size} sx={{ mr: 0.5 }} />
            Sort Desc
          </StyledMenuItem>
        </MenuList>
      </Popover>
    </Box>
  );
};

export default SortingButton;

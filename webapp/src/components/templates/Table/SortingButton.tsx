import { ArrowDownward, ArrowUpward, MoreVert } from '@mui/icons-material';
import { Box, IconButton, MenuItem, MenuList, Popover } from '@mui/material';
import React, { MouseEvent } from 'react';
import { Column, SortModel, State } from './types';

type SortingButtonProps = {
  col: Column<any, any>;
  sortState: SortModel[];
  setState: React.Dispatch<React.SetStateAction<State>>;
  beforeOpen?: (e: MouseEvent) => void;
};

const SortingButton: React.FC<SortingButtonProps> = ({
  col,
  sortState,
  setState,
  beforeOpen,
}) => {
  const [anchorEl, setAnchorEl] = React.useState<HTMLButtonElement | null>(
    null
  );

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
    return sortState.some(
      (item) => item.field === col.field && ordering === item.sort
    );
  };

  const handleSelectSort = (
    col: Column<any, any>,
    ordering: 'asc' | 'desc'
  ) => {
    let alreadySortedByThisField = false;
    const newSortState = sortState.map((item) => {
      if (item.field === col.field) {
        alreadySortedByThisField = true;
        item.sort = ordering;
      }
      return item;
    });
    if (!alreadySortedByThisField) {
      newSortState.push({ field: col.field, sort: ordering });
    }
    setState((prev) => ({ ...prev, sortModel: newSortState }));
  };
  return (
    <Box onMouseDown={beforeOpen}>
      <IconButton
        sx={{
          color: 'gray',
          transition: 'color 0.5s',
        }}
        onClick={handleClick}
      >
        <MoreVert />
      </IconButton>
      <Popover
        anchorOrigin={{
          vertical: 'center',
          horizontal: 'right',
        }}
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
      >
        <MenuList>
          <MenuItem
            selected={isSelectedSorting(col, 'asc')}
            onClick={() => {
              handleSelectSort(col, 'asc');
              handleClose();
            }}
          >
            <ArrowUpward />
            Sort Asc
          </MenuItem>
          <MenuItem
            selected={isSelectedSorting(col, 'desc')}
            onClick={() => {
              handleSelectSort(col, 'desc');
              handleClose();
            }}
          >
            <ArrowDownward />
            Sort Desc
          </MenuItem>
        </MenuList>
      </Popover>
    </Box>
  );
};

export default SortingButton;

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
import React, { MouseEvent } from 'react';
import { Column, SortModel, State } from './types';

interface SortingButtonProps extends Pick<IconButtonProps, 'sx' | 'size'> {
  col: Column<any, any>;
  sortState: SortModel[];
  setState: React.Dispatch<React.SetStateAction<State>>;
  beforeOpen?: (e: MouseEvent) => void;
}

const StyledMenuItem = styled(MenuItem)(({ theme }) => ({
  fontSize: '1rem',
}));

const SortingButton: React.FC<SortingButtonProps> = ({
  col,
  sortState,
  setState,
  beforeOpen,
  sx,
  size = 'small',
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

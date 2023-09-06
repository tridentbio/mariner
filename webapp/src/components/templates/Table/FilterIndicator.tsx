import { FilterAltOffRounded, FilterAltRounded } from '@mui/icons-material';
import { IconButton } from '@mui/material';
import { MouseEventHandler } from 'react';

interface FilterIndicatorProps {
  active?: boolean;
  onClick?: MouseEventHandler<HTMLButtonElement>;
}
const FilterIndicator = (props: FilterIndicatorProps) => {
  return (
    <IconButton onClick={props.onClick} disabled={!props.onClick}>
      <FilterAltRounded sx={{ fontSize: '1rem' }} />
    </IconButton>
  );
};

export default FilterIndicator;

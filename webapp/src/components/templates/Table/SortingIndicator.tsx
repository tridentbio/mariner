import { ArrowDownward, ArrowUpward } from '@mui/icons-material';
import { IconButton } from '@mui/material';

interface SortingIndicatorProps {
  sort: 'asc' | 'desc';
  onClick?: () => void;
}

const SortingIndicator = ({ sort, onClick }: SortingIndicatorProps) => {
  return (
    <IconButton onClick={onClick} disabled={!onClick}>
      {sort === 'asc' ? <ArrowUpward /> : <ArrowDownward />}
    </IconButton>
  );
};

export default SortingIndicator;

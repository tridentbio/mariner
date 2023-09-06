import { ArrowDownward, ArrowUpward } from '@mui/icons-material';
import { IconButton, IconButtonProps } from '@mui/material';

interface SortingIndicatorProps extends Pick<IconButtonProps, 'sx' | 'size'> {
  sort: 'asc' | 'desc';
  onClick?: () => void;
}

const SortingIndicator = ({
  sort,
  onClick,
  size,
  sx,
}: SortingIndicatorProps) => {
  return (
    <IconButton onClick={onClick} disabled={!onClick} size={size} sx={sx}>
      {sort === 'asc' ? (
        <ArrowUpward fontSize={size} />
      ) : (
        <ArrowDownward fontSize={size} />
      )}
    </IconButton>
  );
};

export default SortingIndicator;

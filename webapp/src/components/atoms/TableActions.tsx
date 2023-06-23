import { SxProps } from '@mui/material';
import { Box } from '@mui/system';

export const TableActionsWrapper = ({
  children,
}: {
  children: React.ReactNode;
}) => (
  <Box
    sx={{
      display: 'flex',
      gap: 1,
      alignItems: 'center',
      justifyContent: 'center',
    }}
  >
    {children}
  </Box>
);

export const tableActionsSx: SxProps = {
  position: 'sticky',
  right: -1,
  background: '#f3f3f3',
  textAlign: 'left',
};

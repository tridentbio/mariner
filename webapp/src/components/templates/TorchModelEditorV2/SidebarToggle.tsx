import { ViewSidebarRounded } from '@mui/icons-material';
import { Box, IconButton } from '@mui/material';

export interface SidebarToggleProps {
  onOpen: () => void;
}

export const SidebarToggle = ({ onOpen }: SidebarToggleProps) => {
  const top = 0;

  return (
    <Box
      sx={{
        zIndex: 200,
        position: 'absolute',
        right: 0,
        top: top + 100,
        display: 'flex',
        flexDirection: 'column',
        borderTopLeftRadius: '10px',
        borderBottomLeftRadius: '10px',
        backgroundColor: '#80808033',
        p: 0.6,
      }}
    >
      <IconButton
        data-testid="openOptionsSidebarButton"
        sx={{ p: 1 }}
        color="primary"
        onClick={() => onOpen()}
      >
        <ViewSidebarRounded fontSize="small" />
      </IconButton>
    </Box>
  );
};

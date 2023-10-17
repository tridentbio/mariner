import AppTabs, { AppTabsProps } from '@components/organisms/Tabs';
import { RemoveSharp } from '@mui/icons-material';
import { Box, IconButton } from '@mui/material';
import { ReactNode, useState } from 'react';

export interface SidebarBaseProps {
  header?: React.FC<any>;
  open: boolean;
  onClose?: () => void;
  tabs: AppTabsProps['tabs'];
  onTabChange: AppTabsProps['onChange'];
}

export const SidebarBase = ({
  open,
  header,
  onClose,
  tabs,
  onTabChange,
}: SidebarBaseProps) => {
  const top = 0;
  const sidebarStyleRight = open ? '0' : '-435px';

  return (
    <Box
      sx={{
        position: 'absolute',
        top,
        p: 2,
        pr: 0,
        width: 400,
        height: 'calc(100% - 32px)',
        right: sidebarStyleRight,
        zIndex: 200,
        borderTopLeftRadius: 10,
        borderBottomLeftRadius: 10,
        boxShadow: 'rgba(0,0,0,0.10) -3px 0px 8px',
        transition: 'right 0.6s',
        backgroundColor: 'white',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          flexDirection: 'row',
        }}
      >
        <IconButton color="primary" onClick={() => onClose && onClose()}>
          <RemoveSharp />
        </IconButton>
        {!!header ? header({}) : null}
      </Box>

      <AppTabs
        tabs={tabs}
        boxProps={{
          sx: {
            display: 'flex',
            flexDirection: 'column',
            overflowY: 'auto',
            height: 'calc(100% - 100px)',
          },
        }}
        onChange={onTabChange}
      />
    </Box>
  );
};

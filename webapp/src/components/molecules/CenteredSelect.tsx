import {
  Box,
  MenuItem,
  Select as MuiSelect,
  SelectProps,
  SxProps,
  Typography,
} from '@mui/material';
import { useMemo } from 'react';

const sx: SxProps = {
  width: '100%',
  justifyContent: 'center',
  margin: 'auto',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
};

type Keys = {
  id?: string;
  value?: string;
  children?: string;
};

interface CenteredSelectProps extends SelectProps {
  items: Record<string, any>;
  keys: Keys;
  title: string;
}

const menuItemMapGenerator = (keys: Keys) => {
  const Component = (item: Record<string, any>) => {
    const props = {
      key: item[keys?.id || ''] || item.id,
      value: item[keys?.value || ''] || item.value,
    };
    const children = item[keys?.children || ''] || item.children;

    return <MenuItem {...props}>{children}</MenuItem>;
  };
  Component.displayName = 'MenuItemMap';
  return Component;
};

const Select = ({ items, keys, title, ...props }: CenteredSelectProps) => {
  const menuItemMap = useMemo(() => menuItemMapGenerator(keys), [keys]);

  return (
    <Box sx={sx}>
      <Typography variant="overline" sx={{ fontSize: 18 }}>
        {title}
      </Typography>
      <MuiSelect
        sx={{
          width: '90%',
        }}
        {...props}
      >
        <MenuItem value={-1} style={{ display: 'none' }} />
        {items.map(menuItemMap)}
      </MuiSelect>
    </Box>
  );
};

export default Select;

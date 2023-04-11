import { useState, MouseEvent, ReactElement } from 'react';
import IconButton from 'components/atoms/IconButton';
import { Menu, MenuItem, Tooltip } from '@mui/material';
import { MoreVertRounded } from '@mui/icons-material';
interface Option {
  icon: ReactElement;
  tip: string;
  onClick: () => void;
  key: string;
}
export interface NodeHeaderProps {
  options: Option[];
}
const NodeHeader = (props: NodeHeaderProps) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);
  const handleClick = (event: MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };
  return (
    <div onMouseLeave={handleClose}>
      <IconButton
        id="basic-button"
        aria-controls={open ? 'basic-menu' : undefined}
        aria-haspopup="true"
        aria-expanded={open ? 'true' : undefined}
        onClick={handleClick}
      >
        <MoreVertRounded />
      </IconButton>
      <Menu
        id="basic-menu"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        MenuListProps={{
          'aria-labelledby': 'basic-button',
          onMouseLeave: handleClose,
        }}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
      >
        {props.options.map((opt) => (
          <MenuItem
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
            key={opt.key}
            onClick={opt.onClick}
          >
            <Tooltip title={opt.tip}>{opt.icon}</Tooltip>
          </MenuItem>
        ))}
      </Menu>
    </div>
  );
};
export default NodeHeader;

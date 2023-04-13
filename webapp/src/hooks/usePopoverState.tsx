import React, { useState } from 'react';

const usePopoverState = () => {
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);
  const handleClickOpenPopover = (
    event: React.MouseEvent<HTMLButtonElement>
  ) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };
  const open = Boolean(anchorEl);
  return {
    anchorEl,
    setAnchorEl,
    handleClickOpenPopover,
    handleClose,
    open,
  };
};

export { usePopoverState };

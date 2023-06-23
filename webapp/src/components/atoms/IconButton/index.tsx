import { CSSProperties } from 'react';
import {
  IconButton as MuiIconButton,
  IconButtonProps as MuiIconButtonProps,
} from '@mui/material';
interface IconButtonProps extends MuiIconButtonProps {}
const IconButton = (props: IconButtonProps) => {
  const style: CSSProperties = {
    display: 'flex',
    width: '30px',
    height: '30px',
    justifyContent: 'center',
    alignItems: 'center',
    ...(props.style || {}),
  };
  return <MuiIconButton {...props} style={style} />;
};

export default IconButton;

import { Link as MuiLink, SxProps } from '@mui/material';
import { Theme } from '@mui/system';
import { Link, LinkProps } from 'react-router-dom';

const AppLink = (props: LinkProps & { sx?: SxProps<Theme> }) => {
  return (
    <MuiLink component={Link} {...props}>
      {props.children}
    </MuiLink>
  );
};

export default AppLink;

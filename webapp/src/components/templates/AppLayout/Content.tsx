import { Box, SxProps } from '@mui/material';
import Breadcrumbs from 'components/molecules/Breadcrumbs';
import { ReactNode } from 'react';

const Content = (props: {
  disableBreadcrumb?: boolean;
  children: ReactNode;
  sx?: SxProps;
}) => {
  return (
    <Box
      sx={{
        ml: 'auto',
        mr: 'auto',
        mt: 5,
        mb: 5,
        maxWidth: '80vw',
        ...(props.sx || {}),
      }}
    >
      {!props.disableBreadcrumb && <Breadcrumbs />}
      {props.children}
    </Box>
  );
};

export default Content;

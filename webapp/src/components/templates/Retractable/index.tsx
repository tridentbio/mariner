import { Box } from '@mui/system';
import { ReactNode } from 'react';

const Retractable = ({ children }: { children: ReactNode }) => (
  <Box>{children}</Box>
);

export default Retractable;

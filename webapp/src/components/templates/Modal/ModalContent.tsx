import { Box } from '@mui/system';
import { ReactNode } from 'react';

const ModalContent = (props: { children: ReactNode }) => {
  return <Box sx={{ maxWidth: '80vw' }}>{props.children}</Box>;
};

export default ModalContent;

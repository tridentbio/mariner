import { Box } from '@mui/material';
import { ReactNode } from 'react';

const ModalHeader = ({ children }: { children: ReactNode }) => {
  return (
    <>
      <Box id="title" fontSize={20} marginRight={'auto'}>
        {children}
      </Box>
    </>
  );
};

export default ModalHeader;

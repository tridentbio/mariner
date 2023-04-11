import { ReactNode } from 'react';
import { Modal as MuiModal, Box, IconButton, Divider } from '@mui/material';
import ModalHeader from './ModalHeader';
import ModalContent from './ModalContent';
import { CloseRounded } from '@mui/icons-material';
import { SxProps, SystemStyleObject } from '@mui/system';

const style: SystemStyleObject = {
  boxShadow: '3px 6px rgba(0, 0, 0, 0.25)',
  bgcolor: 'background.paper',
  ml: 'auto',
  mr: 'auto',
  mb: 'auto',
  padding: 2,
  width: '67%',
  maxWidth: 1000,
};

interface ModalProps {
  onClose: () => void;
  open: boolean;
  title?: string;
  children: ReactNode;
  closeOnlyOnClickX?: boolean;
  titleSx?: SxProps;
}
const Modal = (props: ModalProps) => {
  return (
    <MuiModal
      sx={{
        overflowY: 'scroll',
        display: 'flex',
        alignItems: 'center',
        paddingTop: 10,
        paddingBottom: 10,
      }}
      open={props.open}
      onClose={(_e, reason) => {
        if (props.closeOnlyOnClickX && reason === 'backdropClick') {
          return;
        }
        props.onClose();
      }}
      aria-labelledby="modal-modal-title"
      aria-describedby="modal-modal-description"
    >
      <Box id="modal-modal-title" sx={style}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'flex-end',
            mb: 0.5,
          }}
        >
          {props.title && <ModalHeader>{props.title}</ModalHeader>}
          {props.onClose && (
            <IconButton
              sx={{ position: 'inherit', top: 0, right: 1000 }}
              onClick={props.onClose}
            >
              <CloseRounded />
            </IconButton>
          )}
        </Box>
        <Divider sx={{ mb: 0.5 }} />
        <ModalContent>{props.children}</ModalContent>
      </Box>
    </MuiModal>
  );
};

export default Modal;

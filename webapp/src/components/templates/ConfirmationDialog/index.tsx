import {
  Alert,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
} from '@mui/material';
import { useAppDispatch } from 'app/hooks';
import { cleanCurrentDeployment } from 'features/deployments/deploymentsSlice';
import React from 'react';

type ConfirmationDialogProps = {
  title: string;
  text: string;
  alertText?: string;
  open: boolean;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
  handleClickConfirm: (any?: any) => void;
};

const ConfirmationDialog: React.FC<ConfirmationDialogProps> = ({
  open,
  setOpen,
  title,
  handleClickConfirm,
  text,
  alertText,
}) => {
  const dispatch = useAppDispatch();
  const handleCancel = () => {
    setOpen(false);
    dispatch(cleanCurrentDeployment());
  };

  return (
    <Dialog
      open={open}
      onClose={handleCancel}
      aria-labelledby="confirmation-dialog-title"
      aria-describedby="confirmation-dialog-description"
    >
      <DialogTitle id="confirmation-dialog-title">{title}</DialogTitle>
      <DialogContent>
        <DialogContentText id="confirmation-dialog-description">
          {text}
        </DialogContentText>
        {alertText && (
          <Alert variant="outlined" severity="warning" sx={{ mt: 2 }}>
            {alertText}
          </Alert>
        )}
      </DialogContent>
      <Divider />
      <DialogActions>
        <Button variant="contained" onClick={handleCancel}>
          Cancel
        </Button>
        <Button
          variant="contained"
          color="warning"
          onClick={handleClickConfirm}
          autoFocus
        >
          Confirm
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConfirmationDialog;

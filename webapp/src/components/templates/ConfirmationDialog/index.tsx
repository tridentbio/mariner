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
import React from 'react';

type ResultTypes = 'closed' | 'canceled' | 'confirmed';

type ConfirmationDialogProps = {
  title: string;
  text: string;
  alertText?: string;
  open: boolean;
  onResult: (result: ResultTypes) => void;
  setOpen: React.Dispatch<React.SetStateAction<boolean>>;
};

const ConfirmationDialog: React.FC<ConfirmationDialogProps> = ({
  open,
  setOpen,
  title,
  text,
  alertText,
  onResult
}) => {
  const dispatchResult = (result: ResultTypes) => {
    setOpen(false);
    onResult(result);
  }

  return (
    <Dialog
      open={open}
      onClose={() => dispatchResult('closed')}
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
        <Button variant="contained" onClick={() => dispatchResult('canceled')}>
          Cancel
        </Button>
        <Button
          variant="contained"
          color="warning"
          onClick={() => dispatchResult('confirmed')}
          autoFocus
        >
          Confirm
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConfirmationDialog;

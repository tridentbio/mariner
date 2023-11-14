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
  confirmText?: string;
  cancelText?: string;
};

const ConfirmationDialog: React.FC<ConfirmationDialogProps> = ({
  open,
  title,
  text,
  alertText,
  onResult,
  confirmText,
  cancelText,
}) => {
  const dispatchResult = (result: ResultTypes) => {
    onResult(result);
  };

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
          {cancelText || 'Cancel'}
        </Button>
        <Button
          variant="contained"
          color="warning"
          onClick={() => dispatchResult('confirmed')}
          autoFocus
        >
          {confirmText || 'Confirm'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConfirmationDialog;

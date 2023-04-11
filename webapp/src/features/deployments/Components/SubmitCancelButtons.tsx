import { Box, Button } from '@mui/material';
import React from 'react';

type SubmitCancelButtonsProps = {
  onCancel: () => void;
  isNewDeployment: boolean;
};

const SubmitCancelButtons: React.FC<SubmitCancelButtonsProps> = ({
  onCancel,
  isNewDeployment,
}) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'row',
        '& > button': { margin: '0px 16px' },
      }}
    >
      <Button
        variant="contained"
        color="error"
        sx={{ flex: 1 }}
        onClick={onCancel}
      >
        CANCEL
      </Button>

      <Button
        disabled={false}
        variant="contained"
        sx={{ flex: 1 }}
        type="submit"
      >
        {isNewDeployment ? 'CREATE' : 'SAVE'}
      </Button>
    </Box>
  );
};

export default SubmitCancelButtons;

import { Box, CircularProgress, Fab, Modal } from '@mui/material';
import React from 'react';
import SaveIcon from '@mui/icons-material/Save';
import CheckIcon from '@mui/icons-material/Check';
import DownloadingIcon from '@mui/icons-material/Downloading';
import MiscellaneousServicesIcon from '@mui/icons-material/MiscellaneousServices';

type ProcessingModalProps = {
  processing: boolean;
  type?: 'Saving' | 'Fetching' | 'Checking' | 'Processing';
};

const flexCenterStyle = {
  display: 'flex',
  alignItems: 'center',
  justifiContent: 'center',
};

const style = {
  position: 'absolute' as 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  textAlign: 'center',
  ...flexCenterStyle,
  flexDirection: 'column',
};

const ProcessingModal: React.FC<ProcessingModalProps> = ({
  processing,
  type,
}) => {
  const iconVariations = {
    Saving: <SaveIcon />,
    Fetching: <DownloadingIcon />,
    Checking: <CheckIcon />,
    Processing: <MiscellaneousServicesIcon />,
  };

  const icon = type ? iconVariations[type] : undefined;
  const ProgressSx = {
    position: 'absolute',
    top: -5,
    bottom: 0,
    right: 0,
    left: -5,
    zIndex: 1,
  };
  return (
    <Modal
      open={processing}
      aria-labelledby="modal-loading"
      aria-describedby="modal-loading"
    >
      <Box sx={style}>
        <Box sx={{ m: 1, position: 'relative', ...flexCenterStyle }}>
          {!!icon && (
            <Fab aria-label={type} color="primary" size="large">
              {icon}
            </Fab>
          )}
          {processing && (
            <CircularProgress size={66} sx={icon ? ProgressSx : {}} />
          )}
        </Box>
        <span style={{ color: 'white', fontSize: '1rem' }}>{type}</span>
      </Box>
    </Modal>
  );
};

export default ProcessingModal;

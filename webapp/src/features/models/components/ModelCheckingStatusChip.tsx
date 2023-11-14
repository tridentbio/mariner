import { ModelVersion } from '@app/rtk/generated/models';
import { Chip, ChipProps } from '@mui/material';

interface ModelCheckingStatusChipProps extends ChipProps {
  status: ModelVersion['checkStatus'];
}

const ModelCheckingStatusChip = ({
  status,
  ...props
}: ModelCheckingStatusChipProps) => {
  switch (status) {
    case null:
      return <Chip label="Checking" color="warning" {...props} />;
    case 'FAILED':
      return <Chip label="Failed" color="error" {...props} />;
    default:
      return <Chip label="Checked" color="success" {...props} />;
  }
};

export default ModelCheckingStatusChip;

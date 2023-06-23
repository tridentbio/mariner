import { Chip } from '@mui/material';
import React, { useMemo } from 'react';
import { DeploymentStatus } from 'app/rtk/generated/deployments';

interface StatusChipProps extends Record<string, any> {
  status?: DeploymentStatus;
}
type Colors = 'success' | 'error' | 'primary' | undefined;

const StatusChip: React.FC<StatusChipProps> = ({ status, ...rest }) => {
  const colorMap: Record<DeploymentStatus, Colors> = useMemo(
    () => ({
      active: 'success',
      stopped: 'error',
      starting: 'primary',
      idle: undefined,
    }),
    []
  );
  return (
    <Chip
      color={colorMap[status!]}
      label={status}
      {...rest}
      sx={{ textTransform: 'uppercase', ...(rest.sx || {}) }}
    />
  );
};

export default StatusChip;

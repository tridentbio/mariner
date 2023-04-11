import { Chip } from '@mui/material';
import React, { useMemo } from 'react';
import { EDeploymnetStatuses } from '../types';

type StatusChipProps = {
  status: EDeploymnetStatuses;
};
type Colors = 'success' | 'error' | 'primary' | undefined;

const StatusChip: React.FC<StatusChipProps> = ({ status }) => {
  const colorMap: Record<EDeploymnetStatuses, Colors> = useMemo(
    () => ({
      [EDeploymnetStatuses.ACTIVE]: 'success',
      [EDeploymnetStatuses.STOPPED]: 'error',
      [EDeploymnetStatuses.STARTING]: 'primary',
      [EDeploymnetStatuses.IDLE]: undefined,
    }),
    []
  );
  return <Chip color={colorMap[status]} label={status} />;
};

export default StatusChip;

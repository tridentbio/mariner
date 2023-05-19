import { Fab, PropTypes, Tooltip } from '@mui/material';
import { Box } from '@mui/system';
import EditIcon from '@mui/icons-material/Edit';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import ClearIcon from '@mui/icons-material/Clear';
import * as deploymentsApi from 'app/rtk/generated/deployments';
import { useMemo } from 'react';
import { EDeploymnetStatuses } from '../types';
import { useNotifications } from '@app/notifications';

type DeploymentsTableActionsProps = {
  id: number;
  status?: deploymentsApi.DeploymentStatus;
  onClickEdit?: () => void;
  onClickDelete?: (id: number) => void;
};
type Colors = PropTypes.Color | 'success' | 'error' | 'info' | 'warning';

const DeploymentsTableActions: React.FC<DeploymentsTableActionsProps> = ({
  id,
  onClickEdit,
  onClickDelete,
  status,
}) => {
  const [updateDeploy] = deploymentsApi.useUpdateDeploymentMutation();
  const { setMessage } = useNotifications();

  const handleDeployStatus = (id: number, status: 'active' | 'stopped') => {
    updateDeploy({
      deploymentId: id,
      deploymentUpdateInput: { status },
    })
      .unwrap()
      .catch(
        (err) =>
          err.data?.detail &&
          setMessage({ message: err.data.detail, type: 'error' })
      );
  };

  const handleStopDeploy = (id: number) => {
    handleDeployStatus(id, 'stopped');
  };
  const handleStartDeploy = (id: number) => {
    handleDeployStatus(id, 'active');
  };

  const startStopMap = {
    [EDeploymnetStatuses.ACTIVE]: {
      color: 'error',
      Icon: StopIcon,
      onClick: handleStopDeploy,
      tooltip: 'Stop',
    },
    [EDeploymnetStatuses.STOPPED]: {
      color: 'success',
      Icon: PlayArrowIcon,
      onClick: handleStartDeploy,
      tooltip: 'Start',
    },
    [EDeploymnetStatuses.IDLE]: {
      color: 'warning',
      Icon: PlayArrowIcon,
      onClick: handleStartDeploy,
      tooltip: 'Start',
    },
    [EDeploymnetStatuses.STARTING]: undefined,
  };
  const StartStopIcon = useMemo(() => {
    const currentStartStop = startStopMap[status!];
    if (!currentStartStop) return;
    const { color, onClick, Icon, tooltip } = currentStartStop;
    return (
      <Tooltip title={tooltip} placement="top">
        <Fab
          size="small"
          color={color as Colors}
          sx={{ boxShadow: 'none' }}
          aria-label="start/stop"
          onClick={() => onClick(id)}
        >
          <Icon fontSize="inherit" />
        </Fab>
      </Tooltip>
    );
  }, [status]);

  return (
    <Box
      sx={{
        display: 'flex',
        gap: 1,
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {onClickEdit && (
        <Tooltip title={'Edit'} placement="top">
          <Fab
            size="small"
            sx={{
              background: '#384E77',
              boxShadow: 'none',
              '&:hover': { background: '#17294b' },
            }}
            aria-label="edit"
            onClick={() => onClickEdit()}
          >
            <EditIcon fontSize="inherit" sx={{ color: '#fff' }} />
          </Fab>
        </Tooltip>
      )}
      {StartStopIcon}
      {onClickDelete && (
        <Tooltip title="Delete" placement="top">
          <Fab
            size="small"
            sx={{
              boxShadow: 'none',
              '&:hover': { background: '#b3b3b3' },
            }}
            aria-label="delete"
            onClick={() => onClickDelete(id)}
          >
            <ClearIcon fontSize="inherit" />
          </Fab>
        </Tooltip>
      )}
    </Box>
  );
};

export default DeploymentsTableActions;

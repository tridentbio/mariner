import { RemoveRedEyeOutlined } from '@mui/icons-material';
import { MenuItem, Link, IconButton, Tooltip, Typography } from '@mui/material';
import { Box } from '@mui/system';
import { useAppDispatch, useAppSelector } from 'app/hooks';
import {
  MarinerEvent,
  MarinerNotification,
} from 'features/notifications/notificationsAPI';
import { updateEventsAsRead } from 'features/notifications/notificationsSlice';
import { FC } from 'react';
import { flatten } from 'utils';
import DashboardCard from '../DashboardCard';
import NotificationList from '../NotificationList';

interface TrainingCardProps {
  notifications: MarinerNotification[];
}

const TrainingCard: FC<TrainingCardProps> = ({ notifications }) => {
  const updatingEventIds = useAppSelector(
    (state) => state.notifications.updatingEventIds
  );
  const renderEvent = (event: MarinerEvent) => (
    <MenuItem
      disabled={updatingEventIds.includes(event.id)}
      sx={{ display: 'flex', width: '100%' }}
    >
      {event.source === 'training:completed'
        ? event.payload.experiment_name
        : '-'}
    </MenuItem>
  );
  return (
    <DashboardCard
      notifications={notifications}
      url={'/trainings'}
      title="Trainings"
      description="This is where you can find all your trainings"
    >
      <NotificationList
        renderEvent={renderEvent}
        notifications={notifications}
      />
    </DashboardCard>
  );
};

export default TrainingCard;

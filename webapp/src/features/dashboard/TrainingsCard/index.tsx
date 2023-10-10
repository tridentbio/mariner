import { PlayCircle } from '@mui/icons-material';
import { MenuItem } from '@mui/material';
import { useAppSelector } from 'app/hooks';
import {
  MarinerEvent,
  MarinerNotification,
} from 'features/notifications/notificationsAPI';
import { FC } from 'react';
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
      icon={<PlayCircle sx={{ color: 'white' }} fontSize="small" />}
    >
      <NotificationList
        renderEvent={renderEvent}
        notifications={notifications}
      />
    </DashboardCard>
  );
};

export default TrainingCard;

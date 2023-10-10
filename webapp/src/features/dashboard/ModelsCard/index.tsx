import { MarinerNotification } from 'features/notifications/notificationsAPI';
import { FC } from 'react';
import DashboardCard from '../DashboardCard';
import { Schema } from '@mui/icons-material';

interface ModelCardProps {
  notifications: MarinerNotification[];
}

const ModelsCard: FC<ModelCardProps> = ({ notifications }) => {
  return (
    <DashboardCard
      title="Models"
      url="/models"
      notifications={notifications}
      description="This is where you can find your models"
      icon={<Schema sx={{ color: 'white' }} fontSize="small" />}
    ></DashboardCard>
  );
};

export default ModelsCard;

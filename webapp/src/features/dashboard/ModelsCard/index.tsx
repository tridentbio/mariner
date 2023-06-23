import { MarinerNotification } from 'features/notifications/notificationsAPI';
import { FC } from 'react';
import DashboardCard from '../DashboardCard';

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
    ></DashboardCard>
  );
};

export default ModelsCard;

import { MarinerNotification } from 'features/notifications/notificationsAPI';
import { FC } from 'react';
import DashboardCard from '../DashboardCard';

interface CostsCardProps {
  notifications: MarinerNotification[];
}

const CostsCard: FC<CostsCardProps> = ({ notifications }) => {
  return (
    <DashboardCard
      title="Costs"
      url="/#"
      notifications={notifications}
      description="This is where you can find usage statistics the resulting costs"
    />
  );
};

export default CostsCard;

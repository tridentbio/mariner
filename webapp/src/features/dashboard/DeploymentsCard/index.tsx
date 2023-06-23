import { MarinerNotification } from 'features/notifications/notificationsAPI';
import { FC } from 'react';
import DashboardCard from '../DashboardCard';

interface DeploymentsCardProps {
  notifications: MarinerNotification[];
}

const DeploymentsCard: FC<DeploymentsCardProps> = ({ notifications }) => {
  return (
    <DashboardCard
      title="Deployments"
      url="/deployments"
      notifications={notifications}
      description="This is where you can find deployments shared with you."
    ></DashboardCard>
  );
};

export default DeploymentsCard;

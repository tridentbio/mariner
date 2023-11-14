import { MarinerNotification } from 'features/notifications/notificationsAPI';
import { FC } from 'react';
import DashboardCard from '../DashboardCard';
import { CircularProgress, Typography } from '@mui/material';
import AppLink from 'components/atoms/AppLink';
import { useGetMyDatasetsQuery } from 'app/rtk/generated/datasets';
import { Dataset } from '@mui/icons-material';

interface DatasetCardProps {
  notifications: MarinerNotification[];
}

const DatasetCard: FC<DatasetCardProps> = ({ notifications }) => {
  const { data, isLoading } = useGetMyDatasetsQuery({ page: 0, perPage: 1 });
  return (
    <DashboardCard
      notifications={notifications}
      url={'/datasets/'}
      title="Datasets"
      description="This is where you can find your datasets"
      icon={<Dataset sx={{ color: 'white' }} fontSize="small" />}
    >
      {isLoading && <CircularProgress />}
      {data?.total === 0 && (
        <div>
          <Typography variant="body1">
            You dont have any datasets yet...
          </Typography>
          <AppLink to="/datasets#new">Create first dataset </AppLink>
        </div>
      )}
    </DashboardCard>
  );
};

export default DatasetCard;

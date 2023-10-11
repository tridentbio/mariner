import { useAppDispatch, useAppSelector } from 'app/hooks';
import Content from 'components/templates/AppLayout/Content';
import { MarinerNotification } from 'features/notifications/notificationsAPI';
import { useEffect, useMemo } from 'react';
import TrainingCard from './TrainingsCard';
import DatasetsCard from './DatasetsCard';
import ModelsCard from './ModelsCard';
import CostsCard from './CostsCard';
import { Box } from '@mui/system';
import Masonry from '@mui/lab/Masonry';
import { fetchNotificaitions } from 'features/notifications/notificationsSlice';
import ChangeLogCard from './ChangeLogCard';
import DeploymentsCard from './DeploymentsCard';

const bySource = (source: string) => (notification: MarinerNotification) =>
  notification.source === source;
const byTrainingSource = bySource('training:completed');
const byChangelogSource = bySource('changelog');

const Dashboard = () => {
  const dispatch = useAppDispatch();
  const notifications = useAppSelector(
    (state) => state.notifications.notifications
  );
  const notificationsLoading =
    useAppSelector((state) => state.notifications.fetchingNotifications) ===
    'loading';
  const trainingNotifications = useMemo(
    () => notifications.filter(byTrainingSource),
    [notifications]
  );

  const changelogNotifications = useMemo(
    () => notifications.filter(byChangelogSource),
    [notifications]
  );

  useEffect(() => {
    if (!notificationsLoading) {
      dispatch(fetchNotificaitions());
    }
  }, []);

  return (
    <Content>
      <Masonry
        columns={{ sm: 1, md: 2 }}
        spacing={5}
        sx={{ maxWidth: '1500px', margin: 'auto' }}
      >
        {changelogNotifications.length > 0 && (
          <ChangeLogCard notifications={changelogNotifications} />
        )}
        <DatasetsCard notifications={[]} />
        <ModelsCard notifications={[]} />
        <TrainingCard notifications={trainingNotifications} />
        <DeploymentsCard notifications={[]} />
        {/* <CostsCard notifications={[]} /> */}
      </Masonry>
    </Content>
  );
};

export default Dashboard;

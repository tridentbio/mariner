import { Typography } from '@mui/material';
import { Box } from '@mui/system';
import Retractable from 'components/templates/Retractable';
import {
  MarinerEvent,
  ChangelogEvent,
  MarinerNotification,
} from 'features/notifications/notificationsAPI';
import { entries } from 'vega-lite';
import DashboardCard from '../DashboardCard';
import NotificationList from '../NotificationList';
interface ChangeLogCardProps {
  notifications: MarinerNotification[];
}
const isEventChangelog = (event: MarinerEvent): event is ChangelogEvent =>
  event.source === 'changelog';

const ChangeLogCard = (props: ChangeLogCardProps) => {
  const group = <T,>(
    arr: T[],
    fn: (val: T, i: number, arr: T[]) => string | undefined
  ) => {
    const result: { [key: string]: T[] } = {};
    for (let i = 0; i < arr.length; i++) {
      const element = arr[i];
      const key = fn(element, i, arr);
      if (key !== undefined) {
        if (key in result) result[key].push(element);
        else result[key] = [element];
      }
    }
    return result;
  };

  return (
    <DashboardCard
      title="Release Changes"
      notifications={props.notifications}
      url="/#"
      description="..."
    >
      <NotificationList
        renderEvent={(event) => {
          if (isEventChangelog(event)) {
            return (
              <Retractable>
                <Typography variant="h5">{event.payload.version}</Typography>
                <Box>
                  {entries(
                    group(event.payload.changes, (change) => change.type)
                  ).map(([type, changes]) => (
                    <div key={type}>
                      <Typography variant="h6">{type}</Typography>
                      <ul>
                        {changes.map((change, idx) => (
                          <li key={idx}>{change.message}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </Box>
              </Retractable>
            );
          }
        }}
        notifications={props.notifications}
      />
    </DashboardCard>
  );
};

export default ChangeLogCard;

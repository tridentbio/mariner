import { RemoveRedEyeOutlined } from '@mui/icons-material';
import {
  Button,
  IconButton,
  Link,
  List,
  ListItem,
  Tooltip,
} from '@mui/material';
import { useAppDispatch } from 'app/hooks';
import {
  MarinerEvent,
  MarinerNotification,
} from 'features/notifications/notificationsAPI';
import { updateEventsAsRead } from 'features/notifications/notificationsSlice';
import { ReactNode } from 'react';
import { flatten } from 'utils';

interface NotificationListProps {
  notifications: MarinerNotification[];
  renderEvent: (event: MarinerEvent) => ReactNode;
}

const NotificationList = ({
  notifications,
  renderEvent,
}: NotificationListProps) => {
  const dispatch = useAppDispatch();
  const handleMarkAsView = (id: number) => {
    dispatch(updateEventsAsRead([id]));
  };
  const handleMarkAllAsView = (soure: string) => {
    const eventIds = flatten(
      notifications
        .filter((notif) => notif.source === soure)
        .map((notif) => notif.events)
    ).map((event) => event.id);
    dispatch(updateEventsAsRead(eventIds));
  };
  return (
    <>
      {notifications.map((notification, i) => (
        <List
          key={i}
          sx={{
            mt: 1,
            backgroundColor: 'rgba(0, 0, 0, 0.04)',
            borderRadius: '10px',
            display: notification.events.length > 0 ? 'block' : 'none',
            '& *': {
              fontSize: '1rem',
            },
          }}
        >
          {notification.events.map((event, j) => {
            const item = renderEvent(event);

            const itemWithLink = event.url ? (
              <Link
                component="a"
                sx={{
                  textDecoration: 'none',
                  width: '100%',
                }}
                href={event.url}
                key={event.url}
                onClick={(event_) => {
                  event_.stopPropagation();
                  handleMarkAsView(event.id);
                }}
              >
                {item}
              </Link>
            ) : (
              item
            );

            return (
              <ListItem
                key={j}
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                }}
              >
                {itemWithLink}
                <Tooltip title="Mark it as viewed">
                  <IconButton
                    color="primary"
                    onClick={(event_) => {
                      handleMarkAsView(event.id);
                      event_.stopPropagation();
                    }}
                  >
                    <RemoveRedEyeOutlined />
                  </IconButton>
                </Tooltip>
              </ListItem>
            );
          })}
          {notification.events.length > 0 && (
            <ListItem
              onClick={(event) => {
                event.stopPropagation();
                handleMarkAllAsView(notification.source);
              }}
              sx={{ color: 'primary' }}
            >
              <Button
                size="small"
                sx={{
                  borderRadius: 2,
                  paddingX: 2,
                }}
              >
                Mark all as read
              </Button>
            </ListItem>
          )}
        </List>
      ))}
    </>
  );
};

export default NotificationList;

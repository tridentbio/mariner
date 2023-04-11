import { RemoveRedEyeOutlined } from '@mui/icons-material';
import { IconButton, Link, MenuItem, Tooltip, Typography } from '@mui/material';
import { Box } from '@mui/system';
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
      {notifications.map((notification) => (
        <div key={notification.source}>
          {notification.events.map((event) => {
            const item = renderEvent(event);
            const itemWithLink = event.url ? (
              <Link
                component="a"
                style={{ textDecoration: 'none', width: '100%' }}
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
              <Box
                key={event.url}
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  flexDirection: 'row',
                }}
              >
                {itemWithLink}
                <Tooltip title="Mark it as viewed">
                  <div>
                    <IconButton
                      color="primary"
                      onClick={(event_) => {
                        handleMarkAsView(event.id);
                        event_.stopPropagation();
                      }}
                    >
                      <RemoveRedEyeOutlined />
                    </IconButton>
                  </div>
                </Tooltip>
              </Box>
            );
          })}
          {notification.events.length > 0 && (
            <MenuItem
              onClick={(event) => {
                event.stopPropagation();
                handleMarkAllAsView(notification.source);
              }}
              sx={{ marginLeft: 'auto', marginRight: 0, color: 'primary' }}
            >
              <Typography variant="body2">Mark all as read</Typography>
            </MenuItem>
          )}
        </div>
      ))}
    </>
  );
};

export default NotificationList;

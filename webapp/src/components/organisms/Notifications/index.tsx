import { Alert, Snackbar } from '@mui/material';
import { useNotifications } from '../../../app/notifications';

const Notifications = () => {
  const { message, closeMessage } = useNotifications();

  return message ? (
    <Snackbar open={true}>
      <Alert severity={message?.type}>{message.message}</Alert>
    </Snackbar>
  ) : null;
};

export default Notifications;

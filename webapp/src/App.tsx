import { NotificationContextProvider } from './app/notifications';
import useAppNavigation from 'hooks/useAppNavigation';
import Notifications from 'components/organisms/Notifications';
import { Suspense } from 'react';
import { CircularProgress } from '@mui/material';
import { Box } from '@mui/system';
import { WebSocketContextProvider } from 'app/websocket/context';

function App() {
  const { routes } = useAppNavigation();

  return (
    <WebSocketContextProvider>
      <NotificationContextProvider>
        <Notifications />
        <Suspense
          fallback={
            <Box
              sx={{
                color: '#fff',
                width: '100vw',
                height: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <CircularProgress color="primary" />
            </Box>
          }
        >
          {routes}
        </Suspense>
      </NotificationContextProvider>
    </WebSocketContextProvider>
  );
}

export default App;

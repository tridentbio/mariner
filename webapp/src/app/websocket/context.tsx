import {
  createContext,
  FC,
  ReactNode,
  useRef,
  useEffect,
  useCallback,
} from 'react';
import { messageHandler, SocketMessageHandler } from '@app/websocket/handler';
import { useAppDispatch } from 'app/hooks';
import { updateExperiment } from 'features/models/modelSlice';
import * as datasetsApi from 'app/rtk/generated/datasets';
import * as deploymentsApi from 'app/rtk/generated/deployments';
import { useNotifications } from 'app/notifications';
import { updateDataset } from 'features/datasets/datasetSlice';
import { updateDeploymentStatus } from '@features/deployments/deploymentsSlice';
import { useAppSelector } from '@hooks';

const WebSocketContext = createContext<{
  socketHandler: SocketMessageHandler | null;
}>({
  socketHandler: null,
});

export const WebSocketContextProvider: FC<{ children: ReactNode }> = (
  props
) => {
  const socketHandlerRef = useRef<SocketMessageHandler>(messageHandler);
  const dispatch = useAppDispatch();
  const { loggedIn, loginStatus } = useAppSelector((state) => ({
    loggedIn: state.users.loggedIn,
    loginStatus: state.users.loginStatus,
  }));

  const [fetchDatasetById] = datasetsApi.useLazyGetMyDatasetQuery();
  const [fetchDeploymentById] = deploymentsApi.useLazyGetDeploymentQuery();
  const { setMessage } = useNotifications();

  const applyAuthenticatedCallbacks = useCallback(
    (socketHandler: SocketMessageHandler) => {
      socketHandler.on('update-running-metrics', (event) => {
        dispatch(updateExperiment(event.data));
      });
      socketHandler.on('dataset-process-finish', (event) => {
        const dataset = event.data.dataset;
        dispatch(updateDataset(dataset));
        fetchDatasetById({ datasetId: dataset.id });
        setMessage({
          message: event.data.message,
          type: dataset.readyStatus === 'failed' ? 'error' : 'success',
        });
      });
      socketHandler.on('update-deployment', (event) => {
        const updatedData = event.data;
        fetchDeploymentById({ deploymentId: updatedData.deploymentId }, false);
        dispatch(updateDeploymentStatus(updatedData));
      });
    },
    []
  );

  const applyAnonymousCallbacks = useCallback(
    (socketHandler: SocketMessageHandler) => {
      socketHandler.on('update-deployment', (event) => {
        const updatedData = event.data;
        dispatch(updateDeploymentStatus(updatedData));
      });
    },
    []
  );

  useEffect(() => {
    const interval = setInterval(() => {
      if (!socketHandlerRef.current) return;
      if (!socketHandlerRef.current.socket) {
        socketHandlerRef.current.connect();
        applyAnonymousCallbacks(socketHandlerRef.current);
        if (loginStatus === 'idle' && loggedIn) {
          applyAuthenticatedCallbacks(socketHandlerRef.current);
        }
        return;
      }
      if (
        [WebSocket.OPEN, WebSocket.CONNECTING, WebSocket.CLOSING].includes(
          socketHandlerRef.current.socket.readyState
        )
      ) {
        // console.log('Ready State', socketHandlerRef.current?.socket?.readyState)
        // console.log('Socket already connected (or connecting/closing), skipping reconnection')
        return;
      } else {
        socketHandlerRef.current.connect();
        applyAnonymousCallbacks(socketHandlerRef.current);
        if (loginStatus === 'idle' && loggedIn) {
          applyAuthenticatedCallbacks(socketHandlerRef.current);
        }
      }
    }, 1000);
    return () => {
      clearInterval(interval);
    };
  }, [loggedIn, loginStatus]);

  return (
    <WebSocketContext.Provider value={{ socketHandler: messageHandler }}>
      {props.children}
    </WebSocketContext.Provider>
  );
};

export default WebSocketContext;

import {
  createContext,
  FC,
  ReactNode,
  useRef,
  useEffect,
  useCallback,
} from 'react';
import { messageHandler, SocketMessageHandler } from 'app/websocket/types';
import { useAppDispatch } from 'app/hooks';
import { updateExperiment } from 'features/models/modelSlice';
import * as datasetsApi from 'app/rtk/generated/datasets';
import * as deploymentsApi from 'app/rtk/generated/deployments';
import { useNotifications } from 'app/notifications';
import { updateDataset } from 'features/datasets/datasetSlice';
import { updateDeploymentStatus } from '@features/deployments/deploymentsSlice';

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
      return socketHandler;
    },
    []
  );

  const applyAnonymousCallbacks = useCallback(
    (socketHandler: SocketMessageHandler) => {
      socketHandler.on('update-deployment', (event) => {
        const updatedData = event.data;
        dispatch(updateDeploymentStatus(updatedData));
      });
      return socketHandler;
    },
    []
  );

  useEffect(() => {
    let socketHandler = socketHandlerRef.current;
    if (socketHandler.isDisconnected()) socketHandler.connect();

    if (socketHandler.connectionType === 'authenticated') {
      socketHandler = applyAuthenticatedCallbacks(socketHandler);
    } else if (socketHandler.connectionType === 'anonymous') {
      socketHandler = applyAnonymousCallbacks(socketHandler);
    }

    socketHandlerRef.current = socketHandler;

    return () => {
      socketHandler.disconnect();
    };
  }, []);

  return (
    <WebSocketContext.Provider value={{ socketHandler: messageHandler }}>
      {props.children}
    </WebSocketContext.Provider>
  );
};

export default WebSocketContext;

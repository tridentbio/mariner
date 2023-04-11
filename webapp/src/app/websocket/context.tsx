import { createContext, FC, ReactNode, useRef, useEffect } from 'react';
import { messageHandler, SocketMessageHandler } from 'app/websocket/types';
import { useAppDispatch } from 'app/hooks';
import { updateExperiment } from 'features/models/modelSlice';
import * as datasetsApi from 'app/rtk/generated/datasets';
import { useNotifications } from 'app/notifications';
import { updateDataset } from 'features/datasets/datasetSlice';

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
  const { setMessage } = useNotifications();

  useEffect(() => {
    let socketHandler = socketHandlerRef.current;
    if (socketHandler.isDisconnected()) socketHandler.connect();
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

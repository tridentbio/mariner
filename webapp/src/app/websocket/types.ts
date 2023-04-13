import { TOKEN } from '@app/local-storage';
import { User } from '@app/rtk/auth';
import {
  ColumnMeta,
  Dataset,
  DatasetErrors,
  DatasetMetadata,
  DataSummary,
  SplitType,
} from '@app/types/domain/datasets';
import { isDev } from '@utils';

export type UpdateExperiment = {
  type: 'update-running-metrics';
  data: {
    experimentId: number;
    experimentName: string;
    metrics: { [key: string]: number };
    epoch: number;
    stage?: string;
  };
};

export type DatasetProcessed = {
  type: 'dataset-process-finish';
  data: {
    dataset_id: number;
    message: string;
    dataset: Dataset;
  };
};

// Union type for all websocket incoming messages
type MessageType = UpdateExperiment | DatasetProcessed;

type CallbackMap = {
  [MT in MessageType as MT['type']]: (message: MT) => void;
};

if (!import.meta.env.VITE_WS_URL)
  throw new Error(
    'Missing env VITE_WS_URL. Check the .env file if in development or the .env.production during the build process if in production'
  );

export class SocketMessageHandler {
  private address: string;
  socket?: WebSocket;

  callbacks: Partial<CallbackMap>;

  constructor(address = import.meta.env.VITE_WS_URL) {
    this.address = address;
    this.callbacks = {};
    this.setListeners();
  }
  setListeners = () => {
    if (this.socket) {
      this.socket.onerror = (event) => {};
      this.socket.onmessage = (event) => this.handleOnMessage(event);
    }
  };

  isDisconnected = () => {
    return this.socket?.readyState !== WebSocket.OPEN;
  };

  connect = () => {
    const token = localStorage.getItem(TOKEN);
    if (!token) return;
    let access_token: string = '';
    try {
      access_token = JSON.parse(token)?.access_token;
    } catch (err) {
      localStorage.removeItem(TOKEN);
    } finally {
      if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
        this.socket = new WebSocket(`${this.address}?token=${access_token}`);
      }
      this.setListeners();
    }
  };
  disconnect = () => {
    if (this.socket) {
      this.socket.close();
    }
  };

  on = <T extends MessageType['type']>(
    messageType: T,
    callback: CallbackMap[T]
  ) => {
    this.callbacks[messageType] = callback;
    this.setListeners();
  };

  removeListener = <T extends MessageType['type']>(messageType: T) => {
    delete this.callbacks[messageType];
  };

  private handleOnMessage = (event: MessageEvent<string>) => {
    // eslint-disable-next-line
    if (isDev()) console.log('[WEBSOCKET]: ', event.data);
    const data: MessageType = JSON.parse(event.data);
    const eventType = data.type;

    // @ts-ignore
    const callback = this.callbacks[eventType];

    // @ts-ignore
    if (callback) callback(data);
  };
}

export const messageHandler = new SocketMessageHandler();

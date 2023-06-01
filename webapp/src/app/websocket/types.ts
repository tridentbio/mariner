import { DeploymentStatus } from '@app/rtk/generated/deployments';
import { TOKEN } from 'app/local-storage';
import { Dataset } from 'app/types/domain/datasets';
import { isDev } from 'utils';

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

export type UpdateDeployment = {
  type: 'update-deployment';
  data: {
    deploymentId: number;
    status: DeploymentStatus;
  };
};

// Union type for all websocket incoming messages
type MessageType = UpdateExperiment | DatasetProcessed | UpdateDeployment;

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
  connectionType?: 'authenticated' | 'anonymous';
  callbacks: CallbackMap;

  constructor(address = import.meta.env.VITE_WS_URL) {
    this.address = address;
    this.callbacks = {
      'dataset-process-finish': () => {},
      'update-running-metrics': () => {},
      'update-deployment': () => {},
    };
    this.setListeners();
  }
  setListeners = (connectionType?: 'authenticated' | 'anonymous') => {
    if (this.socket) {
      this.socket.onerror = (event) => {
        // eslint-disable-next-line
        console.error('[WEBSOCKET]: ', event);
      };
      this.socket.onmessage = (event) => this.handleOnMessage(event);
      this.connectionType = connectionType;
    }
  };

  isDisconnected = () => {
    return this.socket?.readyState !== WebSocket.OPEN;
  };

  connectAuthenticated = (token: string) => {
    let access_token: string = '';
    try {
      access_token = JSON.parse(token)?.access_token;
    } catch (err) {
      localStorage.removeItem(TOKEN);
    } finally {
      if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
        this.socket = new WebSocket(`${this.address}?token=${access_token}`);
      }
      this.setListeners('authenticated');
    }
  };

  connectAnonymous = (token: string) => {
    if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
      this.socket = new WebSocket(`${this.address}-public?token=${token}`);
    }
    this.setListeners('anonymous');
  };

  connect = () => {
    const authToken = localStorage.getItem(TOKEN);
    if (authToken) return this.connectAuthenticated(authToken);

    const url = window.location.href;
    if (!url.includes('public-model/')) return;

    const publicDeploymentToken = url
      .split('public-model/')[1]
      .split('/')
      .join('.');
    this.connectAnonymous(publicDeploymentToken);
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
    if (data.type === 'update-running-metrics') {
      this.callbacks['update-running-metrics'](data);
    } else if (data.type === 'dataset-process-finish') {
      this.callbacks['dataset-process-finish'](data);
    } else if (data.type === 'update-deployment') {
      this.callbacks['update-deployment'](data);
    }
  };
}

export const messageHandler = new SocketMessageHandler();

import { matchPath } from 'react-router-dom';
import { DeploymentStatus } from '@app/rtk/generated/deployments';
import { ELocalStorage, fetchLocalStorage } from 'app/local-storage';
import { Dataset } from 'app/types/domain/datasets';
import { isDev } from 'utils';
import { Model } from '@app/rtk/generated/models';

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

export type UpdateModel = {
  type: 'update-model';
  data: Model;
};

// Union type for all websocket incoming messages
type MessageType =
  | UpdateExperiment
  | DatasetProcessed
  | UpdateDeployment
  | UpdateModel;

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
      'update-model': () => {},
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
    }
    if (connectionType && !this.connectionType)
      this.connectionType = connectionType;
  };

  isDisconnected = () => {
    return this.socket?.readyState !== WebSocket.OPEN;
  };

  connectAuthenticated = (token: string) => {
    if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
      this.socket = new WebSocket(`${this.address}?token=${token}`);
    }

    this.setListeners('authenticated');
  };

  connectAnonymous = (token: string) => {
    if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
      this.socket = new WebSocket(`${this.address}-public?token=${token}`);
    }
    this.setListeners('anonymous');
  };

  connect = () => {
    const url = window.location.href;
    const publicDeploymentToken = this.getTokenFromPublicDeploymentUrl(url);
    if (publicDeploymentToken) {
      this.connectAnonymous(publicDeploymentToken);
    }

    const authToken = fetchLocalStorage<{ access_token: string }>(
      ELocalStorage.TOKEN
    );
    if (authToken) return this.connectAuthenticated(authToken.access_token);
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

    this.callbacks[data.type](data as any);
  };

  private getTokenFromPublicDeploymentUrl = (url: string) => {
    url = new URL(url).pathname;
    const parsedUrl = matchPath('public-model/:token1/:token2/:token3', url);
    const { token1, token2, token3 } = parsedUrl?.params || {};
    if (!token1 || !token2 || !token3) return null;

    return `${token1}.${token2}.${token3}`;
  };
}

export const messageHandler = new SocketMessageHandler();

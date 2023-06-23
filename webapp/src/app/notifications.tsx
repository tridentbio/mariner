import { AlertColor } from '@mui/material';

import {
  createContext,
  FC,
  ReactElement,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from 'react';

const NOTIF_TIME = 5000;

type MessagePayload = {
  message: string;
  type: AlertColor;
};

type Logger = (msg: string) => void;
interface INotificationContext {
  message?: MessagePayload;
  setMessage: (messagePayload: MessagePayload) => void;
  closeMessage: () => void;
  success: Logger;
  notifyError: Logger;
  warning: Logger;
  info: Logger;
}

const notImplemented = () => {
  throw new Error('not implemented');
};

const NotificationContext = createContext<INotificationContext>({
  setMessage: () => null,
  closeMessage: notImplemented,
  success: notImplemented,
  notifyError: notImplemented,
  warning: notImplemented,
  info: notImplemented,
});

export const NotificationContextProvider: FC<{
  children?: ReactNode;
}> = (props) => {
  const [message, setMessage] = useState<MessagePayload | undefined>();

  useEffect(() => {
    let timeout: NodeJS.Timeout;

    if (message) {
      timeout = setTimeout(() => setMessage(undefined), NOTIF_TIME);
    }

    return () => {
      if (timeout) clearTimeout(timeout);
    };
  }, [message]);

  const error = (msg: string) => {
    setMessage({ type: 'error', message: msg });
  };
  const success = (msg: string) => {
    setMessage({ type: 'success', message: msg });
  };
  const warning = (msg: string) => {
    setMessage({ type: 'warning', message: msg });
  };
  const info = (msg: string) => {
    setMessage({ type: 'info', message: msg });
  };

  return (
    <NotificationContext.Provider
      value={{
        success,
        notifyError: error,
        warning,
        info,
        message,
        setMessage,
        closeMessage: () => setMessage(undefined),
      }}
    >
      {props.children}
    </NotificationContext.Provider>
  );
};

export const useNotifications = (): INotificationContext => {
  const value = useContext(NotificationContext);

  if (!value) {
    throw new Error('useNotifications should be used inside context provider');
  }

  return value;
};

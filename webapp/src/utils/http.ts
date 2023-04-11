import { AxiosError, AxiosResponse } from 'axios';
import { isDev } from 'utils';

export enum Status {
  OK = 200,
  CONFLICT = 409,
  INTERNAL_ERROR = 500,
  BAD_REQUEST = 400,
  UNAUTHORIZED = 401,
}

export class HTTPError extends Error {
  code: number;
  response: AxiosResponse;
  constructor(message: string, code: number, response: AxiosResponse) {
    super(message);
    this.code = code;
    this.response = response;
  }
}

const defaultMessages: {
  [key in Status]: string | null;
} = {
  [Status.INTERNAL_ERROR]: 'Internal Error',
  [Status.UNAUTHORIZED]: 'Unauthorized',
  [Status.OK]: null,
  [Status.CONFLICT]: null,
  [Status.BAD_REQUEST]: null,
};

const makeStatus = (statusCode: number): Status | undefined => {
  if (statusCode in Status) {
    return statusCode as Status;
  }
  if (isDev()) throw new Error('Unexpected statusCode ' + statusCode);
};

export const makeHttpError = (response: AxiosResponse, message?: string) => {
  const status = makeStatus(response.status);

  const defaultMessage = status && defaultMessages[status];
  let msg = '';

  if (message) msg = message;
  else if (defaultMessage) msg = defaultMessage;
  else msg = 'Internal Error';

  return new HTTPError(msg, response.status, response);
};

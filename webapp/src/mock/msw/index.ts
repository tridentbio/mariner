import { rest } from 'msw';
import { fakeApi } from './server';

declare global {
  interface Window {
    msw?: {
      fakeApi: typeof fakeApi
      rest: typeof rest
    }
  }
}

export const startMock = async () => {
  if (window.msw) {
    // eslint-disable-next-line no-console
    console.log('MSW is already running.');
  } else {
    // eslint-disable-next-line no-console
    console.log('MSW has not been started. Starting now.');

    window.msw = { fakeApi, rest };

    await fakeApi.start();
  }
};

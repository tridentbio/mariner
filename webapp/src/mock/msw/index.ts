import { rest } from 'msw';
import { fakeApi } from './server';

export const startMock = async () => {
  if (window.msw) {
    console.log('MSW is already running.');
  } else {
    console.log('MSW has not been started. Starting now.');

    window.msw = { fakeApi, rest };

    await fakeApi.start();
  }
};

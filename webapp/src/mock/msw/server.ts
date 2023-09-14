import { setupWorker } from 'msw';
import { handlers as deployments } from './endpoints/deployments/handlers';
import { handlers as models } from './endpoints/models/handlers';

export const fakeApi = setupWorker(...deployments, ...models);

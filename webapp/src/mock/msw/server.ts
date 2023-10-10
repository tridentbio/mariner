import { setupWorker } from 'msw';
import { handlers as deployments } from './endpoints/deployments/handlers';
import { handlers as models } from './endpoints/models/handlers';
import { handlers as datasets } from './endpoints/datasets/handlers';

export const fakeApi = setupWorker(...deployments, ...models, ...datasets);

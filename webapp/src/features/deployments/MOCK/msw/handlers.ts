import { rest } from 'msw';
import { getDeploymentsData } from '../data';

const baseUrl = `${import.meta.env.VITE_API_BASE_URL}/v1`;
const api = (path: string) => `${baseUrl}${path}`;

export const handlers = [
  rest.get(api('/deployments'), (req, res, ctx) => {
    return res(
      ctx.delay(3000),
      ctx.json({
        data: getDeploymentsData,
        total: 3,
      })
    );
  }),
  rest.get(api('/deployments/*'), (req, res, ctx) => {
    return res(ctx.json(getDeploymentsData[0]));
  }),
  rest.post(api('/deployments/*'), (req, res, ctx) => {
    return res(ctx.status(201), ctx.json(getDeploymentsData[1]));
  }),
  rest.put(api('/deployments/*'), (req, res, ctx) => {
    return res(ctx.json(getDeploymentsData[2]));
  }),
  rest.delete(api('/deployments/*'), (req, res, ctx) => {
    return res(ctx.status(200));
  }),
];

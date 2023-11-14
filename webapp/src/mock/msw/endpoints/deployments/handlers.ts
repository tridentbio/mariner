import { rest } from 'msw';
import { deploymentsData } from './data';
import { api } from '../../api';

export const handlers = [
  rest.get(api('/deployments'), (req, res, ctx) => {
    return res(
      ctx.delay(3000),
      ctx.json({
        data: deploymentsData,
        total: 3,
      })
    );
  }),
  rest.get(api('/deployments/*'), (req, res, ctx) => {
    return res(ctx.json(deploymentsData[0]));
  }),
  rest.post(api('/deployments/*'), (req, res, ctx) => {
    return res(ctx.status(201), ctx.json(deploymentsData[1]));
  }),
  rest.put(api('/deployments/*'), (req, res, ctx) => {
    return res(ctx.json(deploymentsData[2]));
  }),
  rest.delete(api('/deployments/*'), (req, res, ctx) => {
    return res(ctx.status(200));
  }),
];

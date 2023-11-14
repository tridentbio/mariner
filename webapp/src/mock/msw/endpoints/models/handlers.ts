import { rest } from 'msw';
import { losses, modelOptionsData, models } from './data';
import { api } from '../../api';

export const handlers = [
  rest.get(api('/models/'), (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        data: models,
        total: models.length,
      })
    );
  }),
  rest.get(api('/models/options'), (req, res, ctx) => {
    return res(ctx.status(200), ctx.json(modelOptionsData));
  }),
  rest.get(api('/models/name-suggestion'), (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        name: 'test',
      })
    );
  }),
  rest.get(api('/models/losses'), (req, res, ctx) => {
    return res(ctx.status(200), ctx.json(losses));
  }),
  rest.get(api('/models/*'), (req, res, ctx) => {
    return res(ctx.status(200), ctx.json(models[0]));
  }),
];

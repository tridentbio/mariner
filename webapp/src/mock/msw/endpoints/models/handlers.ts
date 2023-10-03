import { rest } from 'msw';
import { modelOptionsData, models } from './data';
import { api } from '../../api';

export const handlers = [
  rest.get(api('/models/?*'), (req, res, ctx) => {
    return res(ctx.status(200), ctx.json(models));
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
];

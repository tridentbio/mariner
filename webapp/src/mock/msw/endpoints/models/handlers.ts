import { rest } from 'msw';
import { modelOptionsData } from './data';
import { api } from '../../api';

export const handlers = [
  rest.get(api('/models/options'), (req, res, ctx) => {
    return res(ctx.status(200), ctx.json(modelOptionsData));
  }),
];

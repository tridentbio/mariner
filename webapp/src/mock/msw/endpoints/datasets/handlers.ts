import { rest } from 'msw';
import { api } from '../../api';
import { datasetsData } from './data';

export const handlers = [
  rest.get(api('/datasets/*'), (req, res, ctx) => {
    return res(
      ctx.delay(500),
      ctx.json({
        data: datasetsData,
        total: datasetsData.length,
      })
    );
  }),
];

import { rest } from 'msw';
import { api } from '../../api';
import { datasetsData, zincCsvMetadata } from './data';

export const handlers = [
  rest.get(api('/datasets'), (req, res, ctx) => {
    return res(
      ctx.delay(100),
      ctx.json({
        data: datasetsData,
        total: datasetsData.length,
      })
    );
  }),
  rest.post(api('/datasets/csv-metadata'), (req, res, ctx) => {
    return res(ctx.delay(200), ctx.json(zincCsvMetadata));
  }),
];

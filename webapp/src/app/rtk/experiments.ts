import { Paginated } from 'app/api';
import {
  Experiment,
  ExperimentHistory,
  FetchExperimentsQuery,
  NewTraining,
} from 'app/types/domain/experiments';
import { enhancedApi } from './generated/experiments';
export const experimentsApi = enhancedApi
  .enhanceEndpoints({ addTagTypes: ['experiments'] })
  .injectEndpoints({
    endpoints: (builder) => ({
      postStartTraining: builder.mutation<Experiment, NewTraining>({
        query: (params) => ({
          url: 'api/v1/experiments/',
          method: 'POST',
          body: params,
        }),
        invalidatesTags: ['experiments'],
      }),
      getExperiments: builder.query<
        Paginated<Experiment>,
        FetchExperimentsQuery
      >({
        query: (params) => ({
          url: 'api/v1/experiments/',
          params,
        }),
        providesTags: (_result, _error, args) => [
          { type: 'experiments', ...args },
        ],
      }),
      getRunningExperiments: builder.query<ExperimentHistory[], void>({
        query: () => 'api/v1/experiments/running-history',
      }),
      cancelTraining: builder.mutation<void, number>({
        query: (id) => ({
          url: `api/v1/experiments/${id}/cancel`,
          method: 'PUT',
        }),
        invalidatesTags: ['experiments'],
      }),
    }),
  });

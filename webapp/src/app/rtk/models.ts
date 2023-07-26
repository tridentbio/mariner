import {
  ModelQuery,
  Model,
  ModelConfig,
  ForwardCheck,
  ModelCreateRequest,
} from 'app/types/domain/models';
import { Paginated } from 'app/api';
import { ModelOptions } from 'app/types/domain/modelOptions';
import { enhancedApi } from './generated/models';
export const modelsApi = enhancedApi.injectEndpoints({
  endpoints: (builder) => ({
    getOptions: builder.query<ModelOptions[], void>({
      query: () => 'api/v1/models/options',
    }),
    getNameSuggestion: builder.query<{ name: string }, void>({
      query: () => 'api/v1/models/name-suggestion',
    }),
    getModelsOld: builder.query<Paginated<Model>, ModelQuery>({
      query: (params) => ({
        url: 'api/v1/models/',
        params,
      }),
    }),
    getModelById: builder.query<Model, number>({
      query: (params) => ({
        url: `api/v1/models/${params}`,
      }),
    }),
    checkModel: builder.mutation<ForwardCheck, ModelConfig>({
      query: (params) => ({
        url: `api/v1/models/check-config`,
        body: params,
        method: 'POST',
      }),
    }),
    createModelOld: builder.mutation<Model, ModelCreateRequest>({
      query: (params) => ({
        url: `api/v1/models/`,
        body: params,
        method: 'POST',
      }),
    }),
    deleteModelOld: builder.mutation<void, number>({
      query: (params) => ({
        url: `api/v1/models/${params}`,
        method: 'DELETE',
      }),
    }),
  }),
});

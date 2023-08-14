// TODO: move to app/rtk

import { Paginated } from 'app/api';
import { api } from 'app/rtk/api';
import {
  Deployment,
  DeploymentWithTrainingData,
} from 'app/rtk/generated/deployments';
import {
  DeploymentCreateRequest,
  DeploymentsQuery,
  DeploymentUpdateRequest,
} from './types';

export const addTagTypes = ['deployments'];

export const deploymentsApi = api
  .enhanceEndpoints({ addTagTypes })
  .injectEndpoints({
    overrideExisting: true,
    endpoints: (builder) => ({
      getDeployments: builder.query<Paginated<Deployment>, DeploymentsQuery>({
        query: () => '/api/v1/deployments/',
        providesTags: ['deployments'],
        keepUnusedDataFor: 3,
      }),
      getDeploymentById: builder.query<DeploymentWithTrainingData, number>({
        query: (deploymentId) => ({
          url: `/api/v1/deployments/${deploymentId}`,
        }),
        providesTags: (_result, _error, arg) => [
          { type: 'deployments', id: arg },
        ],
      }),
      createDeployment: builder.mutation<Deployment, DeploymentCreateRequest>({
        query: (params) => {
          return { url: `/api/v1/deployments/`, body: params, method: 'POST' };
        },
        invalidatesTags: ['deployments'],
      }),
      updateDeployment: builder.mutation<Deployment, DeploymentUpdateRequest>({
        query: ({ deploymentId, ...params }) => ({
          url: `/api/v1/deployments/${deploymentId}`,
          body: params,
          method: 'PUT',
        }),
        invalidatesTags: ['deployments'],
      }),
      deleteDeployment: builder.mutation<void, number>({
        query: (deploymentId) => ({
          url: `/api/v1/deployments/${deploymentId}`,
          method: 'DELETE',
        }),
        invalidatesTags: ['deployments'],
      }),
      startDeployment: builder.mutation<Deployment, number>({
        query: (deploymentId) => ({
          url: `/api/v1/deployments/${deploymentId}/start`,
          method: 'PUT',
        }),
        invalidatesTags: ['deployments'],
      }),
      stopDeployment: builder.mutation<Deployment, number>({
        query: (deploymentId) => ({
          url: `/api/v1/deployments/${deploymentId}/stop`,
          method: 'PUT',
        }),
        invalidatesTags: ['deployments'],
      }),
    }),
  });

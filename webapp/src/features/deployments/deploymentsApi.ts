import { Paginated } from 'app/api';
import { api } from 'app/rtk/api';
import {
  Deployment,
  DeploymentCreateRequest,
  DeploymentsQuery,
  DeploymentUpdateRequest,
} from './types';

export const addTagTypes = ['deployments'];
export const deploymentsApi = api
  .enhanceEndpoints({ addTagTypes })
  .injectEndpoints({
    endpoints: (builder) => ({
      getDeployments: builder.query<Paginated<Deployment>, DeploymentsQuery>({
        query: () => '/deployments/',
        providesTags: ['deployments'],
        keepUnusedDataFor: 3,
      }),
      getDeploymentById: builder.query<Deployment, number>({
        query: (deploymentId) => ({
          url: `/deployments/${deploymentId}`,
        }),
        providesTags: (_result, _error, arg) => [
          { type: 'deployments', id: arg },
        ],
      }),
      createDeployment: builder.mutation<Deployment, DeploymentCreateRequest>({
        query: (params) => ({
          url: `/models/`,
          body: params,
          method: 'POST',
        }),
        invalidatesTags: ['deployments'],
      }),
      updateDeployment: builder.mutation<Deployment, DeploymentUpdateRequest>({
        query: ({ deploymentId, ...params }) => ({
          url: `/models/${deploymentId}`,
          body: params,
          method: 'PUT',
        }),
        invalidatesTags: ['deployments'],
      }),
      deleteDeployment: builder.mutation<void, number>({
        query: (deploymentId) => ({
          url: `/models/${deploymentId}`,
          method: 'DELETE',
        }),
        invalidatesTags: ['deployments'],
      }),
      startDeployment: builder.mutation<Deployment, number>({
        query: (deploymentId) => ({
          url: `/deployments/${deploymentId}/start`,
          method: 'PUT',
        }),
        invalidatesTags: ['deployments'],
      }),
      stopDeployment: builder.mutation<Deployment, number>({
        query: (deploymentId) => ({
          url: `/deployments/${deploymentId}/stop`,
          method: 'PUT',
        }),
        invalidatesTags: ['deployments'],
      }),
    }),
  });

// TODO: move to app/rtk

import { Paginated } from 'app/api';
import {
  Deployment,
  DeploymentWithTrainingData,
  generatedDeploymentsApi,
} from 'app/rtk/generated/deployments';
import {
  DeploymentCreateRequest,
  DeploymentsQuery,
  DeploymentUpdateRequest,
} from './types';

export const addTagTypes = ['deployments'];

export const deploymentsApi = generatedDeploymentsApi
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

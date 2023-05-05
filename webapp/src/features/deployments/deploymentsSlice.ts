import { createSlice } from '@reduxjs/toolkit';
import { RootState } from '../../app/store';
import { DeploymentsState } from './types';
import { Deployment } from 'app/rtk/generated/deployments';
import { deploymentsApi } from './deploymentsApi';

const initialState: DeploymentsState = {
  deployments: [],
  totalDeployments: 0,
};

export const deploymentSlice = createSlice({
  name: 'deployment',
  initialState,
  reducers: {
    setCurrentDeployment: (
      state,
      action: {
        type: string;
        payload: Deployment;
      }
    ) => {
      if (action.payload) {
        state.current = action.payload;
      }
    },
    cleanCurrentDeployment: (state) => {
      state.current = undefined;
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      deploymentsApi.endpoints.getDeployments.matchFulfilled,
      (state, action) => {
        state.deployments = action.payload.data;
        state.totalDeployments = action.payload.total;
      }
    );

    builder.addMatcher(
      deploymentsApi.endpoints.getDeploymentById.matchFulfilled,
      (state, action) => {
        state.current = action.payload;
      }
    );
  },
});

export const selectDeploymentById =
  (deploymentId?: number) => (state: RootState) => {
    if (deploymentId) {
      return state.deployments.deployments.find(
        (deployment) => deployment.id === deploymentId
      );
    }
  };

export const { setCurrentDeployment, cleanCurrentDeployment } =
  deploymentSlice.actions;
export default deploymentSlice.reducer;

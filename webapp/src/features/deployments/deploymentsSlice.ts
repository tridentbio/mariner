import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { RootState } from '../../app/store';
import { DeploymentsState } from './types';
import { Deployment, DeploymentStatus } from 'app/rtk/generated/deployments';
import { generatedDeploymentsApi as deploymentsApi } from '@app/rtk/generated/deployments';

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
    updateDeploymentStatus: (
      state,
      action: PayloadAction<{ deploymentId: number; status: DeploymentStatus }>
    ) => {
      if (state.deployments)
        state.deployments = state.deployments.map((deployment) =>
          deployment.id === action.payload.deploymentId
            ? { ...deployment, status: action.payload.status }
            : deployment
        );
      if (state.current?.id === action.payload.deploymentId)
        state.current.status = action.payload.status;
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
      deploymentsApi.endpoints.getDeployment.matchFulfilled,
      (state, action) => {
        state.current = action.payload;
      }
    );
    builder.addMatcher(
      deploymentsApi.endpoints.getPublicDeployment.matchFulfilled,
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

export const {
  setCurrentDeployment,
  cleanCurrentDeployment,
  updateDeploymentStatus,
} = deploymentSlice.actions;
export default deploymentSlice.reducer;

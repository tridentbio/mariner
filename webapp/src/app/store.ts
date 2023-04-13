import { configureStore, ThunkAction, Action } from '@reduxjs/toolkit';
import datasetsReducer from '@features/datasets/datasetSlice';
import usersReducer from '@features/users/usersSlice';
import modelsReducer from '@features/models/modelSlice';
import notificationsReducer from '@features/notifications/notificationsSlice';
import unitsReducer from '@features/units/unitsSlice';
import deploymentsReducer from '@features/deployments/deploymentsSlice';
import { authApi } from './rtk/auth';
import { datasetsApi } from './rtk/datasets';
import { modelsApi } from './rtk/models';
import { experimentsApi } from './rtk/experiments';
import { api } from './rtk/api';
import { deploymentsApi } from '@features/deployments/deploymentsApi';
import { usersApiRtk } from './rtk/users';

export const store = configureStore({
  reducer: {
    datasets: datasetsReducer,
    users: usersReducer,
    models: modelsReducer,
    deployments: deploymentsReducer,
    notifications: notificationsReducer,
    units: unitsReducer,
    [authApi.reducerPath]: authApi.reducer,
    [datasetsApi.reducerPath]: datasetsApi.reducer,
    [modelsApi.reducerPath]: modelsApi.reducer,
    [experimentsApi.reducerPath]: experimentsApi.reducer,
    [deploymentsApi.reducerPath]: deploymentsApi.reducer,
    [usersApiRtk.reducerPath]: usersApiRtk.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(api.middleware),
});

export type AppDispatch = typeof store.dispatch;

export type RootState = ReturnType<typeof store.getState>;

export type AppThunk<ReturnType = void> = ThunkAction<
  ReturnType,
  RootState,
  unknown,
  Action<string>
>;

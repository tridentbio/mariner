import { configureStore, ThunkAction, Action } from '@reduxjs/toolkit'
import counterReducer from '../features/counter/counterSlice'
import datasetsReducer from '../features/datasets/datasetSlice'
import usersReducer from '../features/users/usersSlice'

export const store = configureStore({
  reducer: {
    counter: counterReducer,
    datasets: datasetsReducer,
    users: usersReducer
  }
})

export type AppDispatch = typeof store.dispatch;
export type RootState = ReturnType<typeof store.getState>;
export type AppThunk<ReturnType = void> = ThunkAction<
  ReturnType,
  RootState,
  unknown,
  Action<string>
>;

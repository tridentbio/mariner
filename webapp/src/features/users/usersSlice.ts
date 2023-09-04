import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { authApi } from 'app/rtk/auth';
import { ELocalStorage, storageSchemas } from '../../app/local-storage';
import * as usersApi from './usersAPI';
import { TablePreferences } from '@components/templates/Table/types';
import { updateStructureByPath } from '@utils';

type Status = 'loading' | 'idle' | 'rejected';
export interface UsersState {
  loggedIn: usersApi.User | null;
  fetchMeStatus: Status;
  loginStatus: Status;
  preferences: {
    tables?: {
      [tableId: string]: TablePreferences;
    };
  };
}

const initialState: UsersState = {
  loggedIn: null,
  fetchMeStatus: 'idle',
  loginStatus: 'idle',
  preferences: {},
};

export const fetchMe = createAsyncThunk('users/fetchMe', async () => {
  const response = await usersApi.getMe();

  return response;
});

export const login = createAsyncThunk(
  'users/login',
  async (payload: { username: string; password: string }) => {
    const response = await usersApi.login(payload.username, payload.password);
    localStorage.setItem(ELocalStorage.TOKEN, JSON.stringify(response));

    return response;
  }
);

export const usersSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    logout: (state) => {
      state.loggedIn = null;
      localStorage.setItem(ELocalStorage.TOKEN, '');
    },
    setPreference: (
      state,
      action: {
        type: string;
        payload: {
          path: string;
          data: { [key: string]: any };
        };
      }
    ) => {
      try {
        const valid = storageSchemas[ELocalStorage.PREFERENCES]?.validate(
          action.payload.data
        );
        if (!valid) throw new Error();

        updateStructureByPath(
          action.payload.path,
          state.preferences,
          action.payload.data
        );

        localStorage.setItem(
          ELocalStorage.PREFERENCES,
          JSON.stringify(state.preferences)
        );
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Invalid user preferences', error);
      }
    },
    loadPreferences: (state) => {
      try {
        const preferences = localStorage.getItem(ELocalStorage.PREFERENCES);

        if (preferences) {
          const parsedPreferences = JSON.parse(preferences);
          const valid =
            storageSchemas[ELocalStorage.PREFERENCES]?.validate(
              parsedPreferences
            );

          if (!valid) throw new Error();

          state.preferences = JSON.parse(preferences);
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Invalid user preferences', error);
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchMe.pending, (state) => {
        state.fetchMeStatus = 'loading';
      })
      .addCase(fetchMe.fulfilled, (state, action) => {
        state.fetchMeStatus = 'idle';
        state.loggedIn = action.payload;
      });
    builder.addCase(fetchMe.rejected, (state) => {
      state.fetchMeStatus = 'rejected';
    });
    builder.addCase(login.pending, (state) => {
      state.loginStatus = 'loading';
    });

    builder.addCase(login.rejected, (state) => {
      state.loginStatus = 'rejected';
    });
    builder.addMatcher(
      authApi.endpoints.login.matchFulfilled,
      (state, action) => {
        localStorage.setItem(
          ELocalStorage.TOKEN,
          JSON.stringify(action.payload)
        );
      }
    );
  },
});

export const { logout, setPreference, loadPreferences } = usersSlice.actions;

export default usersSlice.reducer;

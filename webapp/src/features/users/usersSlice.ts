import { TablePreferences } from '@components/templates/Table/types';
import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { updateStructureByPath } from '@utils';
import { authApi } from 'app/rtk/auth';
import {
  ELocalStorage,
  fetchLocalStorage,
  storageSchemas,
} from '../../app/local-storage';
import * as usersApi from './usersAPI';

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
      state.preferences = {};
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
        if (!state.loggedIn) return;

        let usersPreferences = fetchLocalStorage<{
          [userId: string | number]: UsersState['preferences'];
        }>(ELocalStorage.PREFERENCES);

        if (usersPreferences == null)
          usersPreferences = { [state.loggedIn.id]: {} };
        else if (!usersPreferences[state.loggedIn.id])
          usersPreferences[state.loggedIn.id] = {};

        const currentUserPreferences = usersPreferences[state.loggedIn.id];

        updateStructureByPath(
          action.payload.path,
          currentUserPreferences,
          action.payload.data
        );

        storageSchemas[ELocalStorage.PREFERENCES]?.validate({
          [state.loggedIn?.id]: currentUserPreferences,
        });

        usersPreferences[state.loggedIn.id] = currentUserPreferences;

        localStorage.setItem(
          ELocalStorage.PREFERENCES,
          JSON.stringify(usersPreferences)
        );

        state.preferences = currentUserPreferences;
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Invalid user preferences', error);
      }
    },
    loadPreferences: (state) => {
      try {
        if (!state.loggedIn) return;

        const usersPreferences = fetchLocalStorage<{
          [userId: string | number]: UsersState['preferences'];
        }>(ELocalStorage.PREFERENCES);

        if (!usersPreferences) return;

        const currentUserPreferences = usersPreferences[state.loggedIn.id];

        state.preferences = currentUserPreferences ?? {};
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Invalid user preferences', error);
      }
    },
    mockLogin: (state) => {
      state = {
        ...state,
        loggedIn: {
          email: 'mock@email.com',
          full_name: 'Mock User',
          id: 1,
        },
      };

      return state;
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

export const { logout, setPreference, loadPreferences, mockLogin } =
  usersSlice.actions;

export default usersSlice.reducer;

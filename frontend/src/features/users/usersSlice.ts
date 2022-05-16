import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import * as usersApi from "./usersAPI";

type Status = 'loading' | 'idle' | 'rejected'
export interface UsersState {
  loggedIn: usersApi.User | null,
  fetchMeStatus: Status,
  loginStatus: Status
}

const initialState: UsersState = {
  loggedIn: null,
  fetchMeStatus: 'idle',
  loginStatus: 'idle'
}

export const fetchMe = createAsyncThunk(
  'users/fetchMe',
  async () => {
    const response = await usersApi.getMe()
    return response
  }
)

export const login = createAsyncThunk(
  'users/login',
  async (payload: {username: string, password: string}) => {
    const response = await usersApi.login(payload.username, payload.password)
    return response
  }
)

export const usersSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {},
  extraReducers: builder => {
    builder.addCase(fetchMe.pending, state => {
      state.fetchMeStatus = 'loading'
    })
    .addCase(fetchMe.fulfilled, (state, action) => {
      state.fetchMeStatus = 'idle'
      state.loggedIn = action.payload
    })
    builder.addCase(fetchMe.rejected, state => {
      state.fetchMeStatus = 'rejected'
    })
    builder.addCase(login.pending, state => {
      state.loginStatus = 'loading'
    })
    builder.addCase(login.fulfilled, state => {
      state.loginStatus = 'idle'
    })
    builder.addCase(login.rejected, state => {
      state.loginStatus = 'rejected'
    })
  }
})

export default usersSlice.reducer

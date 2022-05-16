import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import * as usersApi from "./usersAPI";

export interface UsersState {
  loggedIn: usersApi.User | null,
  status: 'loading' | 'idle' | 'rejected'
}

const initialState: UsersState = {
  loggedIn: null,
  status: 'idle'
}

export const fetchme = createAsyncThunk(
  'users/fetchMe',
  async () => {
    const response = await usersApi.getMe()
    return response
  }
)

export const usersSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {},
  extraReducers: builder => {
    builder.addCase(fetchme.pending, state => {
      state.status = 'loading'
    })
    .addCase(fetchme.fulfilled, (state, action) => {
      state.status = 'idle'
      state.loggedIn = action.payload
    })
    builder.addCase(fetchme.rejected, state => {
      state.status = 'rejected'
    })
  }
})

export default usersSlice.reducer

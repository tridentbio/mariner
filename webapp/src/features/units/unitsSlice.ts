import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { Status } from 'app/api';
import * as unitsApi from './unitsAPI';

interface UnitState {
  units: unitsApi.Unit[];
  fetchingUnits: Status;
}
export const fetchUnits = createAsyncThunk(
  'units/fetchUnits',
  unitsApi.fetchUnits
);
const initialState: UnitState = {
  units: [],
  fetchingUnits: 'idle',
};
const unitSlice = createSlice({
  name: 'unit',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder.addCase(fetchUnits.pending, (state) => {
      state.fetchingUnits = 'loading';
    });

    builder.addCase(fetchUnits.fulfilled, (state, action) => {
      state.fetchingUnits = 'idle';
      state.units = action.payload;
    });

    builder.addCase(fetchUnits.rejected, (state) => {
      state.fetchingUnits = 'failed';
    });
  },
});

export default unitSlice.reducer;

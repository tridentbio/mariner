import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { makeS3DataLink } from 'utils';
import api, { Status } from '../../app/api';
import {
  datasetsApi,
  datasetsApi as rtkDatasetApi,
} from '../../app/rtk/datasets';
import { DatasetsListingFilters, Dataset } from 'app/types/domain/datasets';
import { gzipDecompress } from 'utils/gzipCompress';

export interface DatasetState {
  filters: DatasetsListingFilters;
  datasets: Dataset[];
  total: number;
  fetchingDatasets: Status;
  creatingDataset: Status;
  fetchingDataset: Status;
}

const initialState: DatasetState = {
  filters: { page: 0, perPage: 25 },
  fetchingDatasets: 'idle',
  creatingDataset: 'idle',
  fetchingDataset: 'idle',
  total: 0,
  datasets: [],
};

export const datasetSlice = createSlice({
  name: 'datasets',
  initialState,
  reducers: {
    updateDataset: (state, action: PayloadAction<Dataset>) => {
      const i = state.datasets.findIndex((ds) => ds.id === action.payload.id);
      if (i === -1) return;
      state.datasets[i] = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      rtkDatasetApi.endpoints.deleteDataset.matchFulfilled,
      (state, action) => {
        state.datasets = state.datasets.filter(
          (ds) => ds.id !== action.meta.arg.originalArgs
        );
      }
    );
    builder.addMatcher(
      rtkDatasetApi.endpoints.putDataset.matchFulfilled,
      (state, action) => {
        state.datasets = state.datasets.map((ds) =>
          ds.id === action.payload.id ? action.payload : ds
        );
      }
    );
    builder.addMatcher(
      rtkDatasetApi.endpoints.getDatasets.matchPending,
      (state, action) => {
        state.filters = action.meta.arg.originalArgs;
      }
    );
    builder.addMatcher(
      rtkDatasetApi.endpoints.getDatasets.matchFulfilled,
      (state, action) => {
        state.datasets = action.payload.data;
        state.total = action.payload.total;
      }
    );
    builder.addMatcher(
      rtkDatasetApi.endpoints.postDatasets.matchFulfilled,
      (state, action) => {
        state.datasets = [...state.datasets, action.payload];
        state.total += 1;
      }
    );

    builder.addMatcher(
      datasetsApi.endpoints.getMyDatasets.matchFulfilled,
      (state, action) => {
        // @ts-ignore
        state.datasets = action.payload.data;
        state.total = action.payload.total;
      }
    );

    builder.addMatcher(
      datasetsApi.endpoints.getMyDataset.matchFulfilled,
      (state, action) => {
        if (
          state.datasets
            .map((dataset) => dataset.id)
            .includes(action.payload.id)
        ) {
          // @ts-ignore
          state.datasets = state.datasets.map((dataset) =>
            dataset.id === action.payload.id ? action.payload : dataset
          );
        } else {
          // @ts-ignore
          state.datasets.push(action.payload);
        }
      }
    );
  },
});

export const downloadDataset = (
  datasetId: number,
  datasetName: string,
  withError: boolean = false
) => {
  api
    .get(makeS3DataLink(datasetId, withError), { responseType: 'blob' })
    .then(async (response) => {
      const url = URL.createObjectURL(
        await gzipDecompress(response.data as Blob)
      );
      const link = document.createElement('a');

      link.href = url;
      link.setAttribute('download', datasetName);
      document.body.appendChild(link);
      link.click();
    });
};

export const { updateDataset } = datasetSlice.actions;
export default datasetSlice.reducer;

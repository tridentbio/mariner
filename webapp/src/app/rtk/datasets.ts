import { keys, makeForm } from 'utils';
import { Paginated } from 'app/api';
import {
  NewDataset,
  Dataset,
  UpdateDataset,
  ColumnInfo,
  DatasetsListingFilters,
} from 'app/types/domain/datasets';
import { generatedDatasetsApi } from './generated/datasets';

const makeNewDatasetForm = (dataset: Partial<NewDataset>) => {
  const form = new FormData();
  keys(dataset).forEach((key) => {
    if (key === 'columnsMetadata' && typeof dataset[key] !== 'string') {
      const jsonified = JSON.stringify(dataset[key as keyof NewDataset]);
      form.set(key, jsonified);
    } else {
      form.set(key, dataset[key as keyof NewDataset] as string);
    }
  });
  return form;
};

export const addTagTypes = ['datasets', 'utils'] as const;

export const datasetsApi = generatedDatasetsApi
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    overrideExisting: true,
    endpoints: (builder) => ({
      postDatasets: builder.mutation<Dataset, NewDataset>({
        query: (payload) => ({
          url: 'api/v1/datasets/',
          method: 'POST',
          body: makeNewDatasetForm(payload),
        }),
        invalidatesTags: ['datasets'],
      }),
      putDataset: builder.mutation<Dataset, UpdateDataset>({
        query: ({ datasetId, ...payload }) => ({
          url: `api/v1/datasets/${datasetId}`,
          method: 'PUT',
          body: makeNewDatasetForm(payload),
        }),
        invalidatesTags: (_result, _error, arg) => [
          { type: 'datasets', id: arg.datasetId },
        ],
      }),
      getDatasets: builder.query<Paginated<Dataset>, DatasetsListingFilters>({
        query: (params) => ({
          url: 'api/v1/datasets/',
          params,
        }),
        providesTags: ['datasets'],
      }),
      deleteDataset: builder.mutation<Dataset, { datasetId: number }>({
        query: (datasetArg) => ({
          url: `api/v1/datasets/${datasetArg.datasetId}`,
          method: 'DELETE',
        }),
        invalidatesTags: ['datasets'],
      }),
      getDatasetById: builder.query<Dataset, number>({
        query: (datasetId) => `api/v1/datasets/${datasetId}`,
        providesTags: (_result, _error, arg) => [{ type: 'datasets', id: arg }],
      }),
      getColumnsMetadata: builder.mutation<ColumnInfo[], File | Blob>({
        query: (file) => ({
          method: 'POST',
          url: `api/v1/datasets/csv-metadata`,
          body: makeForm({ file }),
        }),
      }),
    }),
  });

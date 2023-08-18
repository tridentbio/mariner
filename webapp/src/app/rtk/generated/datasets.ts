import { api } from '../api';
export const addTagTypes = ['datasets'] as const;
const injectedRtkApi = api
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    endpoints: (build) => ({
      getMyDatasets: build.query<GetMyDatasetsApiResponse, GetMyDatasetsApiArg>(
        {
          query: (queryArg) => ({
            url: `/api/v1/datasets/`,
            params: {
              page: queryArg.page,
              perPage: queryArg.perPage,
              sortByRows: queryArg.sortByRows,
              sortByCols: queryArg.sortByCols,
              sortByCreatedAt: queryArg.sortByCreatedAt,
              searchByName: queryArg.searchByName,
              createdById: queryArg.createdById,
            },
          }),
          providesTags: ['datasets'],
        }
      ),
      createDataset: build.mutation<
        CreateDatasetApiResponse,
        CreateDatasetApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/datasets/`,
          method: 'POST',
          body: queryArg.bodyCreateDatasetApiV1DatasetsPost,
        }),
        invalidatesTags: ['datasets'],
      }),
      getMyDatasetSummary: build.query<
        GetMyDatasetSummaryApiResponse,
        GetMyDatasetSummaryApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/datasets/${queryArg.datasetId}/summary`,
        }),
        providesTags: ['datasets'],
      }),
      getMyDataset: build.query<GetMyDatasetApiResponse, GetMyDatasetApiArg>({
        query: (queryArg) => ({
          url: `/api/v1/datasets/${queryArg.datasetId}`,
        }),
        providesTags: ['datasets'],
      }),
      updateDataset: build.mutation<
        UpdateDatasetApiResponse,
        UpdateDatasetApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/datasets/${queryArg.datasetId}`,
          method: 'PUT',
          body: queryArg.datasetUpdateInput,
        }),
        invalidatesTags: ['datasets'],
      }),
      deleteDataset: build.mutation<
        DeleteDatasetApiResponse,
        DeleteDatasetApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/datasets/${queryArg.datasetId}`,
          method: 'DELETE',
        }),
        invalidatesTags: ['datasets'],
      }),
      getDatasetColumnsMetadata: build.mutation<
        GetDatasetColumnsMetadataApiResponse,
        GetDatasetColumnsMetadataApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/datasets/csv-metadata`,
          method: 'POST',
          body: queryArg.bodyGetDatasetColumnsMetadataApiV1DatasetsCsvMetadataPost,
        }),
        invalidatesTags: ['datasets'],
      }),
    }),
    overrideExisting: false,
  });
export { injectedRtkApi as generatedDatasetsApi };
export type GetMyDatasetsApiResponse =
  /** status 200 Successful Response */ PaginatedDataset;
export type GetMyDatasetsApiArg = {
  page?: number;
  perPage?: number;
  sortByRows?: string;
  sortByCols?: string;
  sortByCreatedAt?: string;
  searchByName?: string;
  createdById?: number;
};
export type CreateDatasetApiResponse =
  /** status 200 Successful Response */ Dataset;
export type CreateDatasetApiArg = {
  bodyCreateDatasetApiV1DatasetsPost: BodyCreateDatasetApiV1DatasetsPost;
};
export type GetMyDatasetSummaryApiResponse =
  /** status 200 Successful Response */ DatasetSummary;
export type GetMyDatasetSummaryApiArg = {
  datasetId: number;
};
export type GetMyDatasetApiResponse =
  /** status 200 Successful Response */ Dataset;
export type GetMyDatasetApiArg = {
  datasetId: number;
};
export type UpdateDatasetApiResponse =
  /** status 200 Successful Response */ Dataset;
export type UpdateDatasetApiArg = {
  datasetId: number;
  datasetUpdateInput: DatasetUpdateInput;
};
export type DeleteDatasetApiResponse =
  /** status 200 Successful Response */ Dataset;
export type DeleteDatasetApiArg = {
  datasetId: number;
};
export type GetDatasetColumnsMetadataApiResponse =
  /** status 200 Successful Response */ ColumnsMeta[];
export type GetDatasetColumnsMetadataApiArg = {
  bodyGetDatasetColumnsMetadataApiV1DatasetsCsvMetadataPost: BodyGetDatasetColumnsMetadataApiV1DatasetsCsvMetadataPost;
};
export type QuantityDataType = {
  domainKind?: 'numeric';
  unit: string;
};
export type NumericalDataType = {
  domainKind?: 'numeric';
};
export type StringDataType = {
  domainKind?: 'string';
};
export type CategoricalDataType = {
  domainKind?: 'categorical';
  classes: {
    [key: string]: number;
  };
};
export type SmileDataType = {
  domainKind?: 'smiles';
};
export type DnaDataType = {
  domainKind?: 'dna';
  isAmbiguous?: boolean;
};
export type RnaDataType = {
  domainKind?: 'rna';
  isAmbiguous?: boolean;
};
export type ProteinDataType = {
  domainKind?: 'protein';
};
export type ColumnsDescription = {
  dataType:
    | QuantityDataType
    | NumericalDataType
    | StringDataType
    | CategoricalDataType
    | SmileDataType
    | DnaDataType
    | RnaDataType
    | ProteinDataType;
  description: string;
  pattern: string;
  datasetId?: number;
};
export type Dataset = {
  id: number;
  name: string;
  description: string;
  rows?: number;
  columns?: number;
  bytes?: number;
  stats?: any;
  dataUrl?: string;
  splitTarget: string;
  splitActual?: string;
  splitType: 'scaffold' | 'random';
  createdAt: string;
  updatedAt: string;
  createdById: number;
  columnsMetadata?: ColumnsDescription[];
  readyStatus?: 'failed' | 'processing' | 'ready';
  errors?: {
    [key: string]: string[] | string;
  };
};
export type PaginatedDataset = {
  data: Dataset[];
  total: number;
};
export type ValidationError = {
  loc: (string | number)[];
  msg: string;
  type: string;
};
export type HttpValidationError = {
  detail?: ValidationError[];
};
export type BodyCreateDatasetApiV1DatasetsPost = {
  name: string;
  description: string;
  splitTarget: string;
  splitType: 'scaffold' | 'random';
  splitOn?: string;
  columnsMetadata?: string;
  file?: Blob;
};
export type DatasetSummary = {
  train: object;
  val: object;
  test: object;
  full: object;
};
export type DatasetUpdateInput = {
  name?: string;
  description?: string;
  splitColumn?: string;
  splitTarget?: string;
  splitType?: 'scaffold' | 'random';
  file?: Blob;
  columnsMetadata?: string;
};
export type ColumnsMeta = {
  name: string;
  dtype?:
    | CategoricalDataType
    | NumericalDataType
    | StringDataType
    | SmileDataType
    | DnaDataType
    | RnaDataType
    | ProteinDataType;
};
export type BodyGetDatasetColumnsMetadataApiV1DatasetsCsvMetadataPost = {
  file?: Blob;
};
export const {
  useGetMyDatasetsQuery,
  useLazyGetMyDatasetsQuery,
  useCreateDatasetMutation,
  useGetMyDatasetSummaryQuery,
  useLazyGetMyDatasetSummaryQuery,
  useGetMyDatasetQuery,
  useLazyGetMyDatasetQuery,
  useUpdateDatasetMutation,
  useDeleteDatasetMutation,
  useGetDatasetColumnsMetadataMutation,
} = injectedRtkApi;

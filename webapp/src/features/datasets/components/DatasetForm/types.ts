import * as datasetsApi from '@app/rtk/generated/datasets';
export type DatasetForm = Omit<
  datasetsApi.BodyCreateDatasetApiV1DatasetsPost,
  'columnsMetadata'
> & {
  columnsMetadata?: datasetsApi.ColumnsDescription[];
};

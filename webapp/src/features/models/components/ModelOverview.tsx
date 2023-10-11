import { FormLabel, Typography } from '@mui/material';
import { Box } from '@mui/system';
// import MDEditor from '@uiw/react-md-editor';
import { Model, ModelColumn } from 'app/types/domain/models';
import { Dataset_ } from 'app/types/domain/datasets';
import AppLink from 'components/atoms/AppLink';
import DataTypeChip from 'components/atoms/DataTypeChip';
import { lazy } from 'react';
import { findColumnMetadata } from 'utils';
import ModelVersions from './ModelVersions';

const TorchModelEditorMarkdown = lazy(
  () => import('utils/codeSplittingAux/TorchModelEditorMarkdown')
);

export interface ModelOverviewProps {
  model: Model;
}
interface ColumnDescriptionProps {
  column: ModelColumn;
  dataset: Dataset_;
}
const ColumnDescription = ({ column, dataset }: ColumnDescriptionProps) => {
  const { columnsMetadata } = dataset;
  const columnMetadata = findColumnMetadata(
    columnsMetadata || [],
    column.columnName
  );
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'row',
        width: '100%',
        alignItems: 'center',
        mb: 0.5,
        fontSize: '1.3rem',
      }}
    >
      {columnMetadata && <DataTypeChip {...columnMetadata.dataType} />}
      <Typography fontWeight="bold" ml={1} fontSize="1.3rem">
        {column.columnName}
      </Typography>
      <Typography ml={1} fontSize="1.3rem">
        column of the
      </Typography>
      <AppLink to={`/datasets/${dataset.id}`} sx={{ ml: 1 }}>
        {dataset.name} dataset
      </AppLink>
    </Box>
  );
};
const ModelOverview = ({ model }: ModelOverviewProps) => {
  const isFeature = (column: ModelColumn) => column.columnType === 'feature';
  const isTarget = (column: ModelColumn) => column.columnType === 'target';
  return (
    <Box>
      {model.description && (
        <>
          <FormLabel>Description</FormLabel>
          <TorchModelEditorMarkdown
            source={model.description}
            wrapperElement={{
              'data-color-mode': 'light',
            }}
          />
        </>
      )}

      <FormLabel>Inputs</FormLabel>
      {model.dataset &&
        model.columns
          .filter(isFeature)
          .map((column) => (
            <ColumnDescription
              column={column}
              dataset={model.dataset!}
              key={column.columnName}
            />
          ))}

      <FormLabel>Target Column</FormLabel>

      {model.dataset &&
        model.columns
          .filter(isTarget)
          .map((column) => (
            <ColumnDescription
              column={column}
              dataset={model.dataset!}
              key={column.columnName}
            />
          ))}

      <FormLabel>Versions</FormLabel>
      <ModelVersions modelId={model.id} versions={model.versions} />
    </Box>
  );
};
export default ModelOverview;

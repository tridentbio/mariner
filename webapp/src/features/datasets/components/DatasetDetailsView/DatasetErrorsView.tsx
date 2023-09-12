import { DatasetErrors } from 'app/types/domain/datasets';
import StackTrace from 'components/organisms/StackTrace';
import { Button } from '@mui/material';
import { LargerBoldText } from '../../../../components/molecules/Text';
import { Box } from '@mui/system';
import { downloadDataset } from 'features/datasets/datasetSlice';

interface DatasetErrorsProps {
  errors: DatasetErrors;
  datasetId: number;
}

const parseMessages = (
  columns: string[],
  rows: string[],
  logs: string[]
): string => {
  let error_str: string = '';

  error_str += 'Columns errors:\n';
  columns.forEach((column: string) => (error_str += '\t' + column + '\n'));

  error_str += '\nRows errors:\n';
  rows.forEach((row: string) => (error_str += '\t' + row + '\n'));

  if (logs.length) {
    error_str += '\nOther logs:\n';
    logs.forEach((log: string) => (error_str += '\t' + log + '\n'));
  }

  error_str += '\nYou can see all errors found downloading the errors dataset';
  return error_str;
};

export const DatasetErrorsView = (props: DatasetErrorsProps) => {
  return (
    <>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingRight: 5,
          mt: 3,
        }}
      >
        <LargerBoldText>Dataset Errors</LargerBoldText>
        <Button
          variant="contained"
          color="primary"
          onClick={() =>
            downloadDataset(
              props.datasetId,
              props.errors.dataset_error_key || 'dataset_with_errors.csv',
              true
            )
          }
        >
          Download dataset with errors
        </Button>
      </Box>
      <StackTrace
        message="Failed to parse dataset (sample)"
        stackTrace={
          parseMessages(
            props.errors.columns || [],
            props.errors.rows || [],
            props.errors.log || []
          ) as string
        }
      />
    </>
  );
};

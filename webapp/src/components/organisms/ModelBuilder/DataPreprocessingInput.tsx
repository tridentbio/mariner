import { Text } from '@components/molecules/Text';
import useModelOptions, {
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { Box } from '@mui/material';
import ColumnsPipelineInput from './ColumnsPipelineInput';
import { DatasetConfigPreprocessing, StepValue } from './types';

export interface DataPreprocessingInputProps {
  value?: DatasetConfigPreprocessing;
}
const DataPreprocessingInput = ({ value }: DataPreprocessingInputProps) => {
  const options = useModelOptions();
  const { featureColumns, targetColumns } = value || {
    featureColumns: [],
    targetColumns: [],
  };

  const preprocessingOptions = options.getPreprocessingOptions();
  const transformOptions = preprocessingOptions
    .filter((option) => option.type === 'transformer')
    .map(toConstructorArgsConfig) as StepValue[];
  const featurizerOptions = preprocessingOptions
    .filter((option) => option.type === 'featurizer')
    .map(toConstructorArgsConfig) as StepValue[];

  return (
    <>
      <Box sx={{ mb: 2, mt: 3 }}>
        <Text variant="subtitle1">Feature Columns:</Text>
        {featureColumns.map((column, index) => (
          <ColumnsPipelineInput
            key={index}
            column={{
              config: column,
              index,
              type: 'featureColumns',
            }}
            featurizerOptions={featurizerOptions}
            transformOptions={transformOptions}
          />
        ))}
      </Box>

      <Box>
        <Text variant="subtitle1">Target Columns:</Text>
        {targetColumns.map((column, index) => (
          <ColumnsPipelineInput
            key={index}
            column={{
              config: column,
              index,
              type: 'targetColumns',
            }}
            featurizerOptions={featurizerOptions}
            transformOptions={transformOptions}
          />
        ))}
      </Box>
    </>
  );
};

export default DataPreprocessingInput;

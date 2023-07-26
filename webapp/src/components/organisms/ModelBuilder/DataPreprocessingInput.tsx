import { Text } from '@components/molecules/Text';
import useModelOptions, {
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { Box } from '@mui/material';
import ColumnsPipelineInput from './ColumnsPipelineInput';
import { DatasetConfigPreprocessing, StepValue } from './types';

export interface DataPreprocessingInputProps {
  value?: DatasetConfigPreprocessing;
  onChange: (value: DatasetConfigPreprocessing) => void;
}
const DataPreprocessingInput = ({
  value,
  onChange,
}: DataPreprocessingInputProps) => {
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
      <Box sx={{ mb: 2 }}>
        <Text variant="h6">Feature Columns:</Text>
        <ColumnsPipelineInput
          value={featureColumns}
          featurizerOptions={featurizerOptions}
          transformOptions={transformOptions}
          onChange={(columns) =>
            onChange({ targetColumns, featureColumns: columns })
          }
        />
      </Box>

      <Box>
        <Text variant="h6">Target Columns:</Text>
        <ColumnsPipelineInput
          value={targetColumns}
          featurizerOptions={featurizerOptions}
          transformOptions={transformOptions}
          onChange={(columns) =>
            onChange({ targetColumns: columns, featureColumns })
          }
        />
      </Box>
    </>
  );
};

export default DataPreprocessingInput;

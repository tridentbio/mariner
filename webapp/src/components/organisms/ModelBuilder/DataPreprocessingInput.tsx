import { ColumnConfig } from '@app/rtk/generated/models';
import { Text } from '@components/molecules/Text';
import ColumnConfigurationAccordion from '@features/models/components/ColumnConfigurationView/ColumnConfigAccordion';
import useModelOptions, {
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import PreprocessingStepSelect from './PreprocessingStepSelect';
import { StepValue } from './types';

type GenericPreprocessingStep = {
  type: string;
  constructorArgs: object;
};
type SimpleColumnConfig = {
  name: string;
  dataType: ColumnConfig['dataType'];
  featurizers: GenericPreprocessingStep[];
  transforms: GenericPreprocessingStep[];
};
type DatasetConfigPreprocessing = {
  featureColumns: SimpleColumnConfig[];
  targetColumns: SimpleColumnConfig[];
};
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

  if ('error' in options) {
    // temporary error handling
    return <div>Failed to load options</div>;
  }
  const preprocessingOptions = options.getPreprocessingOptions();
  const transformOptions = preprocessingOptions
    .filter((option) => option.type === 'transformer')
    .map(toConstructorArgsConfig);
  const featurizerOptions = preprocessingOptions
    .filter((option) => option.type === 'featurizer')
    .map(toConstructorArgsConfig);

  return (
    <>
      {featureColumns.map((column) => (
        <ColumnConfigurationAccordion
          key={column.name}
          dataType={column.dataType}
          name={column.name}
        >
          <Text sx={{ width: '100%' }}>Featurizers:</Text>
          {column.featurizers.map((step) => (
            <PreprocessingStepSelect
              options={featurizerOptions}
              onChange={console.log}
              key={step.type}
            />
          ))}
          <PreprocessingStepSelect
            options={featurizerOptions}
            onChange={console.log}
          />

          <Text sx={{ width: '100%' }}>Transformers:</Text>
          {column.transforms.map((step) => (
            <PreprocessingStepSelect
              options={transformOptions}
              onChange={console.log}
              key={step.type}
            />
          ))}
          <PreprocessingStepSelect
            options={transformOptions}
            onChange={console.log}
          />
        </ColumnConfigurationAccordion>
      ))}

      {targetColumns.map((column) => (
        <ColumnConfigurationAccordion
          key={column.name}
          dataType={column.dataType}
          name={column.name}
        ></ColumnConfigurationAccordion>
      ))}
    </>
  );
};
export default DataPreprocessingInput;

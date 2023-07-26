import { ColumnConfig } from '@app/rtk/generated/models';
import { Text } from '@components/molecules/Text';
import ColumnConfigurationAccordion from '@features/models/components/ColumnConfigurationView/ColumnConfigAccordion';
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
  value: DatasetConfigPreprocessing;
}
const DataPreprocessingInput = ({
  value: { featureColumns, targetColumns },
}: DataPreprocessingInputProps) => {
  const preprocessingOptions: StepValue[] = [];
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
              options={preprocessingOptions}
              onChange={console.log}
              key={step.type}
            />
          ))}

          <Text sx={{ width: '100%' }}>Transformers:</Text>
          {column.transforms.map((step) => (
            <PreprocessingStepSelect
              options={preprocessingOptions}
              onChange={console.log}
              key={step.type}
            />
          ))}
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

import { DataTypeGuard } from '@app/types/domain/datasets';
import { Text } from '@components/molecules/Text';
import ColumnConfigurationAccordion from '@features/models/components/ColumnConfigurationView/ColumnConfigAccordion';
import DeleteOutline from '@mui/icons-material/DeleteOutline';
import { Button, IconButton } from '@mui/material';
import { useFieldArray, useFormContext } from 'react-hook-form';
import PreprocessingStepSelect from './PreprocessingStepSelect';
import {
  DatasetConfigPreprocessing,
  GenericPreprocessingStep,
  SimpleColumnConfig,
  StepValue,
} from './types';

export interface ColumnsPipelineInputProps {
  column: {
    config: SimpleColumnConfig;
    index: number;
    type: 'featureColumns' | 'targetColumns';
  };
  featurizerOptions: StepValue[];
  transformOptions: StepValue[];
}

export default function ColumnsPipelineInput(props: ColumnsPipelineInputProps) {
  const { featurizerOptions, transformOptions, column } = props;
  const { control } = useFormContext<DatasetConfigPreprocessing>();

  const transformsOptionsField = useFieldArray({
    control,
    name: `${column.type}.${column.index}.transforms`,
  });

  const addTransform = () => {
    transformsOptionsField.append({
      type: '',
      constructorArgs: {},
    } as GenericPreprocessingStep);
  };

  const deleteColumnTransform = (stepIndex: number) => {
    transformsOptionsField.remove(stepIndex);
  };

  return (
    <>
      <ColumnConfigurationAccordion
        key={column.config.name}
        dataType={column.config.dataType}
        name={column.config.name}
      >
        {!DataTypeGuard.isNumericalOrQuantity(column.config.dataType) && (
          <>
            <Text sx={{ width: '100%' }}>Featurizers:</Text>
            <PreprocessingStepSelect
              stepFieldName={`${column.type}.${column.index}.featurizers.0`}
              options={featurizerOptions}
            />
          </>
        )}

        <Text sx={{ width: '100%' }}>Transforms:</Text>
        {transformsOptionsField.fields.map((step, stepIndex) => (
          <PreprocessingStepSelect
            key={step.id}
            stepFieldName={`${column.type}.${column.index}.transforms.${stepIndex}`}
            options={transformOptions}
            extra={
              <IconButton onClick={() => deleteColumnTransform(stepIndex)}>
                <DeleteOutline />
              </IconButton>
            }
          />
        ))}
        <Button
          variant="contained"
          sx={{ mt: 1 }}
          onClick={() => addTransform()}
        >
          ADD
        </Button>
      </ColumnConfigurationAccordion>
    </>
  );
}

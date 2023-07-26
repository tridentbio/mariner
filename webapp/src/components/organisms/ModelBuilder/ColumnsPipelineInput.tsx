import { Text } from '@components/molecules/Text';
import ColumnConfigurationAccordion from '@features/models/components/ColumnConfigurationView/ColumnConfigAccordion';
import DeleteOutline from '@mui/icons-material/DeleteOutline';
import { Button, IconButton } from '@mui/material';
import PreprocessingStepSelect from './PreprocessingStepSelect';
import { SimpleColumnConfig, StepValue } from './types';
import { DataTypeGuard } from '@app/types/domain/datasets';

export interface ColumnsPipelineInputProps {
  onChange: (value: SimpleColumnConfig[]) => void;
  value: SimpleColumnConfig[];
  featurizerOptions: StepValue[];
  transformOptions: StepValue[];
}

export default function ColumnsPipelineInput(props: ColumnsPipelineInputProps) {
  const { featurizerOptions, transformOptions, onChange, value } = props;
  const updateStepsArray = (
    steps: StepValue[],
    stepIndex: number,
    value: StepValue
  ) => {
    return steps.map((step, index) => {
      if (index === stepIndex) {
        return value;
      }
      return step;
    });
  };

  const updateColumnTransform = (
    columns: SimpleColumnConfig[],
    columnIndex: number,
    stepIndex: number,
    value: StepValue
  ) => {
    return (columns || []).map((column, index) => {
      if (index === columnIndex) {
        return {
          ...column,
          transforms: updateStepsArray(column.transforms, stepIndex, value),
        };
      }
      return column;
    });
  };

  const updateColumnFeaturizer = (
    columns: SimpleColumnConfig[],
    columnIndex: number,
    stepIndex: number,
    value: StepValue
  ) => {
    return (columns || []).map((column, index) => {
      if (index === columnIndex) {
        return {
          ...column,
          featurizers: updateStepsArray(column.featurizers, stepIndex, value),
        };
      }
      return column;
    });
  };
  const addTransform = (
    columns: SimpleColumnConfig[],
    columnIndex: number,
    value: StepValue | undefined
  ) => {
    return columns.map((column, index) => {
      if (index === columnIndex) {
        return {
          ...column,
          transforms: [...column.transforms, value],
        };
      }
      return column;
    });
  };

  const deleteColumnTransform = (
    columns: SimpleColumnConfig[],
    columnIndex: number,
    stepIndex: number
  ) => {
    return columns.map((column, index) => {
      if (index === columnIndex) {
        return {
          ...column,
          transforms: column.transforms.filter(
            (_, index) => index !== stepIndex
          ),
        };
      }
      return column;
    });
  };
  return (
    <>
      {value.map((column, columnIndex) => (
        <ColumnConfigurationAccordion
          key={column.name}
          dataType={column.dataType}
          name={column.name}
        >
          {!DataTypeGuard.isNumericalOrQuantity(column.dataType) && (
            <>
              <Text sx={{ width: '100%' }}>Featurizers:</Text>
              <PreprocessingStepSelect
                options={featurizerOptions}
                onChange={(step) =>
                  onChange(updateColumnFeaturizer(value, columnIndex, 0, step))
                }
              />
            </>
          )}

          <Text sx={{ width: '100%' }}>Transforms:</Text>
          {column.transforms.map((step, stepIndex) => (
            <PreprocessingStepSelect
              options={transformOptions}
              onChange={(newStep) =>
                onChange(
                  updateColumnTransform(value, columnIndex, stepIndex, newStep)
                )
              }
              value={step}
              key={step?.type || 'new' + stepIndex}
              extra={
                <IconButton
                  onClick={() =>
                    onChange(
                      deleteColumnTransform(value, columnIndex, stepIndex)
                    )
                  }
                >
                  <DeleteOutline />
                </IconButton>
              }
            />
          ))}
          <Button
            variant="contained"
            sx={{ mt: 1 }}
            onClick={() =>
              onChange(addTransform(value, columnIndex, undefined))
            }
          >
            ADD
          </Button>
        </ColumnConfigurationAccordion>
      ))}
    </>
  );
}

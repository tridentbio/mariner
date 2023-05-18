import { ColumnConfig } from '@app/rtk/generated/models';
import { DataTypeDomainKind } from '@app/types/domain/datasets';
import {
  CategoricalInput,
  QuantityInput,
  SmileInput,
  StringInput,
} from '@features/models/components/ModelVersionInferenceView/ModelInput/inputs';
import { BiologicalInput } from '@features/models/components/ModelVersionInferenceView/ModelInput/inputs/BiologicalInput';
import { Alert, Box } from '@mui/material';
import { useCallback } from 'react';

type InferenceInputProps = {
  inferenceColumn: ColumnConfig;
  value: string | number;
  handleChange: (value: string | number) => void;
};

export const InferenceInput = ({
  inferenceColumn,
  handleChange,
  value,
}: InferenceInputProps) => {
  const commonProps = <T extends any>() => ({
    onChange: handleChange,
    value: value as T,
    label: inferenceColumn.name,
    key: inferenceColumn.name,
  });

  const getBioInput = useCallback(
    (domainKind: string) => (
      <BiologicalInput
        {...commonProps<string>()}
        domainKind={domainKind as 'dna' | 'rna' | 'protein'}
      />
    ),
    [inferenceColumn.dataType]
  );

  const NotFound = <Alert color="error" key={inferenceColumn.name} />;

  if (!inferenceColumn.dataType || !inferenceColumn.dataType.domainKind)
    return NotFound;

  return (
    {
      [DataTypeDomainKind.Smiles]: <SmileInput {...commonProps<string>()} />,

      [DataTypeDomainKind.Numerical]: (
        <QuantityInput
          {...commonProps<number>()}
          unit={
            'unit' in inferenceColumn.dataType
              ? inferenceColumn.dataType.unit
              : ''
          }
          value={value as number}
        />
      ),

      [DataTypeDomainKind.Categorical]: (
        <CategoricalInput
          {...commonProps<string>()}
          getLabel={(value) => value as string}
          options={
            'classes' in inferenceColumn.dataType
              ? Object.keys(inferenceColumn.dataType.classes!)
              : []
          }
        />
      ),

      [DataTypeDomainKind.String]: <StringInput {...commonProps<string>()} />,

      [DataTypeDomainKind.Dna]: getBioInput(
        inferenceColumn.dataType.domainKind
      ),

      [DataTypeDomainKind.Rna]: getBioInput(
        inferenceColumn.dataType.domainKind
      ),

      [DataTypeDomainKind.Protein]: getBioInput(
        inferenceColumn.dataType.domainKind
      ),
    }[inferenceColumn.dataType.domainKind] || NotFound
  );
};

type InferenceInputsProps = {
  inferenceColumns: ColumnConfig[];
  values: {
    [key: string]: string | number;
  };
  handleChange: (key: string, value: string | number) => void;
};

export const InferenceInputs = ({
  inferenceColumns,
  values,
  handleChange,
}: InferenceInputsProps) => (
  <>
    {inferenceColumns.map((inferenceColumn) => (
      <Box sx={{ mb: '1rem' }}>
        <InferenceInput
          key={inferenceColumn.name}
          inferenceColumn={inferenceColumn}
          value={values[inferenceColumn.name]}
          handleChange={(value: string | number) =>
            handleChange(inferenceColumn.name, value)
          }
        />
      </Box>
    ))}
  </>
);

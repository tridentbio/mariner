import {
  FormControl,
  FormControlProps,
  InputLabel,
  MenuItem,
  Select,
  SelectProps,
} from '@mui/material';
import UnitAutocomplete from 'components/molecules/UnitsAutocomplete';
import { DataTypeDomainKind, DataTypeGuard } from 'app/types/domain/datasets';
import { Control, Controller, useFormContext } from 'react-hook-form';
import { useState } from 'react';
import { required } from 'utils/reactFormRules';
import { DatasetForm } from '../types';
import { DataType } from '@model-compiler/src/interfaces/torch-model-editor';

type EventLike<T> = {
  target: {
    value: T;
  };
};
interface SelectDomainKindProps
  extends Omit<SelectProps<DataType['domainKind']>, 'onChange' | 'value'> {
  onChange?: (event: EventLike<DataType['domainKind'] | undefined>) => any;
  value?: DataType['domainKind'];
}

const SelectDomainKind = (props: SelectDomainKindProps) => {
  return (
    <Select
      {...props}
      onChange={(event) => {
        return (
          props.onChange &&
          props.onChange(event as EventLike<DataType['domainKind']>)
        );
      }}
      value={props.value}
    >
      <MenuItem
        value={DataTypeDomainKind.Categorical}
        key={DataTypeDomainKind.Categorical}
      >
        Categorical
      </MenuItem>
      <MenuItem value={DataTypeDomainKind.Dna} key={DataTypeDomainKind.Dna}>
        DNA
      </MenuItem>
      <MenuItem
        value={DataTypeDomainKind.Numerical}
        key={DataTypeDomainKind.Numerical}
      >
        Numeric
      </MenuItem>
      <MenuItem
        value={DataTypeDomainKind.Protein}
        key={DataTypeDomainKind.Protein}
      >
        Protein
      </MenuItem>
      <MenuItem value={DataTypeDomainKind.Rna} key={DataTypeDomainKind.Rna}>
        RNA
      </MenuItem>
      <MenuItem
        value={DataTypeDomainKind.Smiles}
        key={DataTypeDomainKind.Smiles}
      >
        SMILES
      </MenuItem>
      <MenuItem
        value={DataTypeDomainKind.String}
        key={DataTypeDomainKind.String}
      >
        String
      </MenuItem>
    </Select>
  );
};

interface DataTypeProps extends Omit<FormControlProps, 'onChange' | 'value'> {
  value?: DataType;
  label: string;
  onChange?: (event: EventLike<DataType>) => any;
  error?: boolean;
  index?: number;
  control?: Control<DatasetForm, any>;
  pattern?: string;
}
const DataTypeInput = ({
  value,
  label,
  onChange,
  error,
  index,
  control,
  pattern,
  ...formProps
}: DataTypeProps) => {
  const [domainKindValue, setDomainKindValue] = useState<
    DataType['domainKind'] | undefined
  >(value?.domainKind);
  const [unitValue, setUnitValue] = useState<string>(
    (value && DataTypeGuard.isQuantity(value) && value.unit) || ''
  );
  const form = useFormContext();
  const domainKind = form.watch(`columnsMetadata.${index}.dataType.domainKind`);
  return (
    <FormControl sx={{ width: '100%' }} {...formProps}>
      <InputLabel id="select-label">{label}</InputLabel>
      {index !== undefined ? (
        <>
          <Controller
            control={control}
            name={`columnsMetadata.${index}.dataType.domainKind`}
            render={({ field, fieldState: { error } }) => (
              <SelectDomainKind
                {...field}
                value={field.value || undefined}
                data-testid={`data-type-input-${pattern}`}
                error={!!error}
                label={error?.message || `Data Type ${index}`}
              />
            )}
          />
          {domainKind === DataTypeDomainKind.Numerical && (
            <Controller
              control={control}
              name={`columnsMetadata.${index}.dataType.unit`}
              rules={{ ...required }}
              render={({
                field,
                formState: {
                  errors: { columnsMetadata: colsError },
                },
                fieldState: { error },
              }) => (
                <UnitAutocomplete
                  pattern={pattern}
                  error={
                    error?.message ||
                    (colsError && colsError[index]?.dataType?.message)
                  }
                  value={{ name: field.value }}
                  onChange={(unit) =>
                    field.onChange({ target: { value: unit?.name || '' } })
                  }
                />
              )}
            />
          )}
        </>
      ) : (
        <>
          <SelectDomainKind
            label={`Data Type`}
            onChange={(event) => setDomainKindValue(event.target.value)}
            value={domainKindValue}
            data-testid="dataset-col-data-type"
          />

          <UnitAutocomplete
            label={'Unit'}
            value={{ name: unitValue }}
            onChange={(unit) => setUnitValue(unit?.name || '')}
            pattern={pattern}
          />
        </>
      )}
    </FormControl>
  );
};

export default DataTypeInput;

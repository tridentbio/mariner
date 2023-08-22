import {
  forwardRef,
  ReactNode,
  Ref,
  useEffect,
  useImperativeHandle,
  useMemo,
  useState,
} from 'react';
import { Alert } from '@mui/material';
import {
  ColumnMeta,
  DataTypeDomainKind,
  isBioType,
} from 'app/types/domain/datasets';
import { Box } from '@mui/system';
import {
  SmileInput,
  QuantityInput,
  StringInput,
  CategoricalInput,
} from './inputs';
import { findMeta } from '../utils';
import { ModelConfig, ModelInputValue } from 'app/types/domain/models';
import { BiologicalInput } from './inputs/BiologicalInput';

interface ModelInputProps {
  columnsMeta: ColumnMeta[];
  columns: string[];
  config: ModelConfig;
  // TODO: persist this in the database
  onChange: (inputs: ModelInputValue) => any;
  value?: ModelInputValue;
}
interface ModelInputHandle {
  reset: () => void;
}
interface ModelInputItemProps {
  children: ReactNode;
}
const ModelInputItem = ({ children }: ModelInputItemProps) => {
  return (
    <Box sx={{ display: 'flex', mb: 1, alignItems: 'center' }}>{children}</Box>
  );
};

const ModelInput = forwardRef(
  (
    { onChange, columns, columnsMeta, config }: ModelInputProps,
    ref: Ref<ModelInputHandle>
  ) => {
    const isFeatureColumn = (colName: string): boolean => {
      const featureColumns = config.dataset.featureColumns;
      if (!featureColumns) return false;
      return featureColumns.map((col) => col.name).includes(colName);
    };

    const initialValues = useMemo<ModelInputValue>(() => {
      const result: ModelInputValue = {};
      for (const key of columns) {
        if (isFeatureColumn(key)) result[key] = [''];
      }
      return result;
    }, [config]);

    const [value, setValue] = useState<ModelInputValue>(initialValues);

    useEffect(() => {
      setValue(initialValues);
    }, [initialValues]);

    const handleChange = (key: string, fieldValue: number | string) => {
      const newValue: ModelInputValue = {
        ...value,
        [key]: [fieldValue] as string[] | number[],
      };
      setValue(newValue);
      onChange && onChange(newValue);
    };

    useImperativeHandle(ref, () => ({
      reset: () => {
        setValue(initialValues);
      },
    }));

    // Sorted columns in decreasing size of pattern
    // is a quick way to fix some ambiguous column descriptions
    // (multiple descriptions for the same column)
    const sortedColumnsMeta = useMemo(() => {
      const arr = [...columnsMeta];
      arr.sort((a, b) => (a.pattern.length < b.pattern.length ? 1 : -1));
      return arr;
    }, [columnsMeta]);
    return (
      <>
        {columns.filter(isFeatureColumn).map((col) => {
          const meta = findMeta(col, sortedColumnsMeta);
          if (!meta) {
            return (
              <Alert color="error" key={col}>
                Missing data type for column {col}
              </Alert>
            );
          }
          const dataType = meta.dataType;
          let children: ReactNode;

          const fieldValue = value[col];

          if (!fieldValue) return null;

          if (dataType.domainKind === DataTypeDomainKind.Smiles) {
            children = (
              <SmileInput
                key={col}
                onChange={(value) => handleChange(col, value)}
                value={fieldValue[0] as string}
                label={col}
              />
            );
          } else if (
            dataType.domainKind === DataTypeDomainKind.Numerical &&
            'unit' in dataType
          ) {
            children = (
              <QuantityInput
                label={col}
                unit={dataType.unit}
                onChange={(value) => handleChange(col, value)}
                value={fieldValue[0] as number}
              />
            );
          } else if (dataType.domainKind === DataTypeDomainKind.Categorical) {
            children = (
              <CategoricalInput
                getLabel={(value) => value as string}
                value={fieldValue[0]}
                onChange={(newVal) => handleChange(col, newVal)}
                options={Object.keys(dataType.classes)}
                key={col}
                label={col}
              />
            );
          } else if (dataType.domainKind === DataTypeDomainKind.String) {
            children = (
              <StringInput
                onChange={(newVal) => handleChange(col, newVal)}
                value={''}
                key={col}
                label={col}
              />
            );
          } else if (isBioType(dataType.domainKind as DataTypeDomainKind)) {
            children = (
              <BiologicalInput
                onChange={(newVal) => handleChange(col, newVal)}
                value={fieldValue[0] as string}
                key={col}
                label={col}
                domainKind={
                  dataType.domainKind as
                    | DataTypeDomainKind.Dna
                    | DataTypeDomainKind.Rna
                    | DataTypeDomainKind.Protein
                }
              />
            );
          } else {
            throw new Error(`Unexpected data type "${dataType}"`);
          }
          if (children)
            return <ModelInputItem key={col}>{children}</ModelInputItem>;
          else return null;
        })}
      </>
    );
  }
);

ModelInput.displayName = 'ModelInput';

export default ModelInput;

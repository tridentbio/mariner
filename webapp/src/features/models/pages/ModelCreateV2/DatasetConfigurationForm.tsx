import {
  DatasetConfig,
  ModelCreate,
  TorchDatasetConfig,
} from '@app/rtk/generated/models';
import { Section } from '@components/molecules/Section';
import DataPreprocessingInput from '@components/organisms/ModelBuilder/DataPreprocessingInput';
import {
  DatasetConfigPreprocessing,
  SimpleColumnConfig,
} from '@components/organisms/ModelBuilder/types';
import ColumnConfigurationView from '@features/models/components/ColumnConfigurationView';
import { FormColumns } from '@features/models/components/ColumnConfigurationView/types';
import { yupResolver } from '@hookform/resolvers/yup';
import { unwrapDollar } from '@model-compiler/src/utils';
import { Box } from '@mui/material';
import { useEffect, useMemo } from 'react';
import {
  Control,
  Controller,
  FormProvider,
  useForm,
  useFormContext,
  useWatch,
} from 'react-hook-form';

export type GenericTransform = {
  name: string;
  constructorArgs: Record<string, any>;
  fowardArgs: Record<string, string | string[]>;
  type: string;
};

const isTransform = (transform: GenericTransform) => {
  const transformerType = transform.name.split('-').at(0) as
    | 'transform'
    | 'featurizer';
  return transformerType === 'transform';
};

const stages = ['col', 'featurizers', 'transforms'] as const;

/* const groupColumnsTransformsFeaturizers = ({
  datasetConfig,
}: {
  datasetConfig: TorchDatasetConfig | DatasetConfig;
}) => {
  const formColumns = {
    feature: datasetConfig.featureColumns.map((col) => ({
      col,
      transforms: [],
      featurizers: [],
    })),
    target: datasetConfig.targetColumns.map((col) => ({
      col,
      transforms: [],
      featurizers: [],
    })),
  } as FormColumns;

  const transformers = [
    ...(datasetConfig.transforms || []),
    ...(datasetConfig.featurizers || []),
  ].sort((a, b) => {
    const aPos = parseInt(a.name!.split('-').at(-1) || '0');
    const bPos = parseInt(b.name!.split('-').at(-1) || '0');
    return aPos - bPos;
  });

  const findTransformerColumn = (
    transformer: GenericTransform,
    formColumns: FormColumns,
    stage: 'col' | 'transforms' | 'featurizers',
    colType: 'feature' | 'target' = 'feature'
  ): ['feature' | 'target' | null, number] => {
    const name = unwrapDollar(
      Object.values(transformer.fowardArgs)[0]! as string
    );
    const colIndex = formColumns[colType].findIndex((col) => {
      const component = col[stage]!;
      if (Array.isArray(component)) {
        return component.some((c) => c.name === name);
      }

      return component.name === name;
    });

    if (colIndex !== -1) return [colType, colIndex];
    else if (colType === 'feature')
      return findTransformerColumn(transformer, formColumns, stage, 'target');
    else return [null, -1];
  };

  transformers.forEach((transformer) => {
    if (!transformer.name) return;

    let [colType, colIndex]: ['feature' | 'target' | null, number] = [null, -1];

    for (const stage of stages) {
      [colType, colIndex] = findTransformerColumn(
        transformer as any as GenericTransform,
        formColumns,
        stage
      );

      if (colIndex !== -1) break;
    }
    if (colIndex === -1) throw new Error('Column not found');

    const transformerType = isTransform(transformer as any as GenericTransform)
      ? 'transforms'
      : 'featurizers';

    // @ts-ignore
    formColumns[colType!][colIndex][transformerType]!.push(transformer);
  });

  return formColumns;
}; */

export const DatasetConfigurationForm = () => {
  const { control } = useFormContext<ModelCreate>();

  return (
    <Box>
      <Section title="Data Configuration">
        <Controller
          control={control}
          name="config.dataset"
          render={({ field }) => (
            <DataPreprocessingInput
              value={{
                featureColumns: field.value
                  .featureColumns as SimpleColumnConfig[],
                targetColumns: field.value
                  .targetColumns as SimpleColumnConfig[],
              }}
            />
          )}
        />
      </Section>
    </Box>
  );
};

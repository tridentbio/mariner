import { Section } from '@components/molecules/Section';
import { Control, useWatch } from 'react-hook-form';
import {
  ColumnConfig,
  DatasetConfig,
  ModelCreate,
  TorchDatasetConfig,
} from '@app/rtk/generated/models';
import { unwrapDollar } from '@model-compiler/src/utils';
import { useMemo } from 'react';
import ColumnConfigurationView from '@features/models/components/ColumnConfigurationView';

export type Transformer = {
  name: string;
  constructorArgs: Record<string, any>;
  fowardArgs: Record<string, null>;
  type: string;
};

type Transforms = Transformer[];

type Featurizers = Transformer[];

type FormColumns = Record<
  'feature' | 'target',
  {
    col: ColumnConfig;
    transforms: Transforms;
    featurizers: Featurizers;
  }[]
>;

const isTransform = (transform: Transformer) => {
  const transformerType = transform.name.split('-').at(0) as
    | 'transform'
    | 'featurizer';
  return transformerType === 'transform';
};

const stages = ['col', 'featurizers', 'transforms'] as const;

const groupColumnsTransformsFeaturizers = ({
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
    transformer: Transformer,
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
        transformer as any as Transformer,
        formColumns,
        stage
      );

      if (colIndex !== -1) break;
    }
    if (colIndex === -1) throw new Error('Column not found');

    const transformerType = isTransform(transformer as any as Transformer)
      ? 'transforms'
      : 'featurizers';

    // @ts-ignore
    formColumns[colType!][colIndex][transformerType]!.push(transformer);
  });

  return formColumns;
};

interface DatasetConfigurationProps {
  control: Control<ModelCreate>;
  setValue: (name: string, value: any) => void;
}
export const DatasetConfigurationForm = ({
  control,
  setValue,
}: DatasetConfigurationProps) => {
  const datasetConfig = useWatch({
    control,
    name: 'config.dataset',
  });

  const formColumns = useMemo(
    () =>
      groupColumnsTransformsFeaturizers({
        datasetConfig,
      }),
    [datasetConfig]
  );

  return (
    <>
      <Section title="Data Configuration">
        <ColumnConfigurationView
          datasetConfig={datasetConfig}
          formColumns={formColumns}
          setValue={setValue}
        />
      </Section>
    </>
  );
};

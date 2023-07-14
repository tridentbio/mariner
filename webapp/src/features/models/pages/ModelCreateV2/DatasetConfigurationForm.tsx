import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
} from '@mui/material';
import { Section } from '@components/molecules/Section';
import { Control, useWatch, useForm } from 'react-hook-form';
import {
  ColumnConfig,
  DatasetConfig,
  ModelCreate,
  TorchDatasetConfig,
} from '@app/rtk/generated/models';
import { Text } from '@components/molecules/Text';
import { DataTypeGuard } from '@app/types/domain/datasets';
import { unwrapDollar } from '@model-compiler/src/utils';
import { useMemo, useState } from 'react';
import { ArrayElement } from '@utils';
import { AddTransformerModal } from '@features/models/components/AddTransformerModal';
import { Session } from 'inspector';
import { TransformerConstructorForm } from '@features/models/components/TransformerConstructorForm';

interface DatasetConfigurationProps {
  control: Control<ModelCreate>;
  setValue: (name: string, value: any) => void;
}

interface ColumnConfigurationAccordionProps {
  name: string;
  dataType: ColumnConfig['dataType'];
  textProps?: Record<string, any>;
  children: React.ReactNode;
}

const reprDataType = (dataType: ColumnConfig['dataType']) => {
  if (DataTypeGuard.isQuantity(dataType))
    return `(${dataType.domainKind}, ${dataType.unit})`;
  else if (DataTypeGuard.isSmiles(dataType)) return `(SMILES)`;
  return `(${dataType.domainKind})`;
};

const ColumnConfigurationAcordion = ({
  name,
  dataType,
  textProps = {},
  children,
}: ColumnConfigurationAccordionProps) => {
  return (
    <Accordion>
      <AccordionSummary>
        <Text {...textProps}>
          {name} {reprDataType(dataType)}
        </Text>
      </AccordionSummary>
      <AccordionDetails>{children}</AccordionDetails>
    </Accordion>
  );
};

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

interface ColumnConfigurationProps {
  formColumn: FormColumns['feature'][0];
  addTransformer: (
    transform: Transformer,
    transformerGroup?: 'transforms' | 'featurizers'
  ) => void;
  control: Control<ModelCreate>;
}

const ColumnConfiguration = ({
  formColumn,
  control,
  addTransformer,
}: ColumnConfigurationProps) => {
  const [openTransformModal, setOpenTransformModal] = useState(false);
  const [openFeaturizerModal, setOpenFeaturizerModal] = useState(false);

  const { col, transforms, featurizers } = formColumn;

  return (
    <>
      <Section title="Featurizers">
        {featurizers.map((transform) => (
          <TransformerConstructorForm
            key={transform.name}
            transformer={transform}
            control={control}
          />
        ))}
      </Section>
      <Section title="Transforms">
        {transforms.map((transform) => (
          <TransformerConstructorForm
            key={transform.name}
            transformer={transform}
            control={control}
          />
        ))}
      </Section>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          margin: 'auto',
          gap: '1rem',
        }}
      >
        {!DataTypeGuard.isNumericalOrQuantity(col.dataType) && (
          <Button
            variant="outlined"
            color="primary"
            sx={{ width: '30%' }}
            onClick={() => setOpenFeaturizerModal(true)}
          >
            Add Featurizer
          </Button>
        )}
        <Button
          variant="outlined"
          color="primary"
          sx={{ width: '30%' }}
          onClick={() => setOpenTransformModal(true)}
        >
          Add Transform
        </Button>
        <AddTransformerModal
          open={openTransformModal}
          cancel={() => setOpenTransformModal(false)}
          confirm={(transform) => {
            addTransformer(transform);
            setOpenTransformModal(false);
          }}
        />
        <AddTransformerModal
          open={openFeaturizerModal}
          cancel={() => setOpenFeaturizerModal(false)}
          confirm={(transform) => {
            addTransformer(transform, 'featurizers');
            setOpenFeaturizerModal(false);
          }}
          transfomerType="featurizer"
        />
      </Box>
    </>
  );
};

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

export const DatasetConfigurationForm = ({
  control,
  setValue,
}: DatasetConfigurationProps) => {
  const datasetConfig = useWatch({
    control,
    name: 'config.dataset',
  });

  const addTransformerFunction =
    (col: ArrayElement<FormColumns['feature']>) =>
    (
      component: Transformer,
      transformerGroup: 'transforms' | 'featurizers' = 'transforms'
    ) => {
      const transformers = [col.col, ...col.featurizers, ...col.transforms];
      const lastTransform = transformers.at(-1)!;

      const fowardArgs = Object.fromEntries(
        Object.entries(component.fowardArgs).map(([key, _]) => [
          key,
          `$${lastTransform.name}`,
        ])
      );
      const newTransformer = {
        ...component,
        name: `${transformerGroup.replace(/s$/g, '')}-${component.name}-${
          transformers.length
        }`,
        fowardArgs,
      };

      setValue(`config.dataset.${transformerGroup}`, [
        ...(datasetConfig[transformerGroup] || []),
        newTransformer,
      ]);
    };

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
        {formColumns.feature.map((formColumn) => (
          <ColumnConfigurationAcordion
            name={formColumn.col.name}
            dataType={formColumn.col.dataType}
          >
            <ColumnConfiguration
              control={control}
              formColumn={formColumn}
              addTransformer={addTransformerFunction(formColumn)}
            />
          </ColumnConfigurationAcordion>
        ))}

        {formColumns.target.map((formColumn) => (
          <ColumnConfigurationAcordion
            name={formColumn.col.name}
            dataType={formColumn.col.dataType}
            textProps={{ fontWeight: 'bold' }}
          >
            <ColumnConfiguration
              control={control}
              formColumn={formColumn}
              addTransformer={addTransformerFunction(formColumn)}
            />
          </ColumnConfigurationAcordion>
        ))}
      </Section>
    </>
  );
};

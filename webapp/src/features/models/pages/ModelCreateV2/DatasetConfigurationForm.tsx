import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
} from '@mui/material';
import { Section } from '@components/molecules/Section';
import { Control, useWatch } from 'react-hook-form';
import {
  ColumnConfig,
  DatasetConfig,
  ModelCreate,
  TorchDatasetConfig,
} from '@app/rtk/generated/models';
import { DataTypeGuard } from '@app/types/domain/datasets';
import { Text } from '@components/molecules/Text';
import { unwrapDollar } from '@model-compiler/src/utils';
import { useMemo } from 'react';

interface DatasetConfigurationProps {
  control: Control<ModelCreate>;
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

type Transforms = DatasetConfig['transforms'];

type Featurizers = DatasetConfig['featurizers'];

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
}

const ColumnConfiguration = ({ formColumn }: ColumnConfigurationProps) => {
  const { col, transforms, featurizers } = formColumn;

  return (
    <>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          margin: 'auto',
          gap: '1rem',
        }}
      >
        {!DataTypeGuard.isNumericalOrQuantity(col.dataType) && (
          <Button variant="outlined" color="primary" sx={{ width: '30%' }}>
            Add Featurizer
          </Button>
        )}
        <Button variant="outlined" color="primary" sx={{ width: '30%' }}>
          Add Transform
        </Button>
      </Box>
    </>
  );
};

const isTransform = (_: any) => {
  return true;
};

const stages = ['col', 'transforms', 'featurizers'] as const;

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
  ];

  const findTransformerColumn = (
    transformer: (typeof transformers)[0],
    formColumns: FormColumns,
    stage: 'col' | 'transforms' | 'featurizers',
    colType: 'feature' | 'target' = 'feature'
  ): ['feature' | 'target' | null, number] => {
    const name = unwrapDollar(transformer.name as string);

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
        transformer,
        formColumns,
        stage
      );

      if (colIndex !== -1) break;
    }

    const transformerType = isTransform(transformer)
      ? 'transforms'
      : 'featurizers';

    formColumns[colType!][colIndex][transformerType]!.push(transformer);
  });

  return formColumns;
};

export const DatasetConfigurationForm = ({
  control,
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
        {formColumns.feature.map((formColumn) => (
          <ColumnConfigurationAcordion
            name={formColumn.col.name}
            dataType={formColumn.col.dataType}
          >
            <ColumnConfiguration formColumn={formColumn} />
          </ColumnConfigurationAcordion>
        ))}

        {formColumns.target.map((formColumn) => (
          <ColumnConfigurationAcordion
            name={formColumn.col.name}
            dataType={formColumn.col.dataType}
            textProps={{ fontWeight: 'bold' }}
          >
            <ColumnConfiguration formColumn={formColumn} />
          </ColumnConfigurationAcordion>
        ))}
      </Section>
    </>
  );
};

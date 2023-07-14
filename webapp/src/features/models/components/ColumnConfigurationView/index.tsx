import { Box, Button } from '@mui/material';
import { ColumnConfig, DatasetConfig } from '@app/rtk/generated/models';
import { DataTypeGuard } from '@app/types/domain/datasets';
import { useState } from 'react';
import { ArrayElement } from '@utils';
import { AddTransformerModal } from '@features/models/components/AddTransformerModal';
import { TransformerConstructorForm } from '@features/models/components/TransformerConstructorForm';
import { CustomAccordion } from '@components/molecules/CustomAccordion';

const reprDataType = (dataType: ColumnConfig['dataType']) => {
  if (DataTypeGuard.isQuantity(dataType))
    return `(${dataType.domainKind}, ${dataType.unit})`;
  else if (DataTypeGuard.isSmiles(dataType)) return `(SMILES)`;
  return `(${dataType.domainKind})`;
};

interface ColumnConfigurationAccordionProps {
  name: string;
  dataType: ColumnConfig['dataType'];
  textProps?: Record<string, any>;
  children: React.ReactNode;
}

const ColumnConfigurationAccordion = ({
  name,
  dataType,
  textProps = {},
  children,
}: ColumnConfigurationAccordionProps) => {
  return (
    <CustomAccordion
      title={`${name} ${reprDataType(dataType)}`}
      textProps={textProps}
      children={children}
    />
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
}

const ColumnConfiguration = ({
  formColumn,
  addTransformer,
}: ColumnConfigurationProps) => {
  const [openTransformModal, setOpenTransformModal] = useState(false);
  const [openFeaturizerModal, setOpenFeaturizerModal] = useState(false);

  const { col, transforms, featurizers } = formColumn;

  return (
    <Box
      sx={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        alignItems: 'center',
      }}
    >
      <CustomAccordion
        title="Featurizers"
        sx={{
          minWidth: '70%',
        }}
      >
        {featurizers.map((transform) => (
          <TransformerConstructorForm
            key={transform.name}
            transformer={transform}
          />
        ))}
      </CustomAccordion>
      <CustomAccordion
        title="Transforms"
        sx={{
          minWidth: '70%',
        }}
      >
        {transforms.map((transform) => (
          <TransformerConstructorForm
            key={transform.name}
            transformer={transform}
          />
        ))}
      </CustomAccordion>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'flex-start',
          margin: 'auto',
          gap: '1rem',
          minWidth: '70%',
        }}
      >
        {!DataTypeGuard.isNumericalOrQuantity(col.dataType) && (
          <Button
            variant="outlined"
            color="primary"
            sx={{ width: '20%', padding: '1rem', fontSize: '20px' }}
            onClick={() => setOpenFeaturizerModal(true)}
          >
            Add Featurizer
          </Button>
        )}
        <Button
          variant="outlined"
          color="primary"
          sx={{ width: '20%', padding: '1rem', fontSize: '20px' }}
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
    </Box>
  );
};

const ColumnConfigurationView = ({
  formColumns,
  datasetConfig,
  setValue,
}: {
  formColumns: FormColumns;
  datasetConfig: DatasetConfig;
  setValue: (k: string, v: any) => void;
}) => {
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

  const deleteTransformerFunction =
    (col: ArrayElement<FormColumns['feature']>) =>
    (
      component: Transformer,
      transformerGroup: 'transforms' | 'featurizers' = 'transforms'
    ) => {
      const transformers = [col.col, ...col.featurizers, ...col.transforms];
      const index = transformers.findIndex((t) => t.name === component.name);

      if (index === -1) return;

      transformers.splice(index, 1);

      setValue(`config.dataset.${transformerGroup}`, transformers);
    };

  return (
    <>
      {formColumns.feature.map((formColumn) => (
        <ColumnConfigurationAccordion
          name={formColumn.col.name}
          dataType={formColumn.col.dataType}
        >
          <ColumnConfiguration
            formColumn={formColumn}
            addTransformer={addTransformerFunction(formColumn)}
          />
        </ColumnConfigurationAccordion>
      ))}

      {formColumns.target.map((formColumn) => (
        <ColumnConfigurationAccordion
          name={formColumn.col.name}
          dataType={formColumn.col.dataType}
          textProps={{ fontWeight: 'bold' }}
        >
          <ColumnConfiguration
            formColumn={formColumn}
            addTransformer={addTransformerFunction(formColumn)}
          />
        </ColumnConfigurationAccordion>
      ))}
    </>
  );
};

export default ColumnConfigurationView;

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
  ModelCreate,
  TargetConfig,
} from '@app/rtk/generated/models';
import { Text } from '@components/molecules/Text';
import { DataTypeGuard } from '@app/types/domain/datasets';

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

interface ColumnConfigurationProps {
  column: ColumnConfig | TargetConfig;
}

const ColumnConfiguration = ({ column }: ColumnConfigurationProps) => {
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
        {!DataTypeGuard.isNumericalOrQuantity(column.dataType) && (
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

export const DatasetConfigurationForm = ({
  control,
}: DatasetConfigurationProps) => {
  const featureColumns = useWatch({
    control,
    name: 'config.dataset.featureColumns',
  });
  const targetColumns = useWatch({
    control,
    name: 'config.dataset.targetColumns',
  });

  console.log({ values: control._formValues });

  return (
    <>
      <Section title="Data Configuration">
        {featureColumns.map((col) => (
          <ColumnConfigurationAcordion name={col.name} dataType={col.dataType}>
            <ColumnConfiguration column={col} />
          </ColumnConfigurationAcordion>
        ))}

        {targetColumns.map((col) => (
          <ColumnConfigurationAcordion
            name={col.name}
            dataType={col.dataType}
            textProps={{ fontWeight: 'bold' }}
          >
            <ColumnConfiguration column={col} />
          </ColumnConfigurationAcordion>
        ))}
      </Section>
    </>
  );
};

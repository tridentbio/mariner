import { DataTypeGuard } from '@app/types/domain/datasets';
import NoData from '@components/atoms/NoData';
import { CustomAccordion } from '@components/molecules/CustomAccordion';
import { Text } from '@components/molecules/Text';
import PreprocessingStepSelect from '@components/organisms/ModelBuilder/PreprocessingStepSelect';
import { Button, Link } from '@mui/material';
import { Box } from '@mui/system';
import { useState } from 'react';
import { TransformConstructorForm } from '../TransformConstructorForm';
import { FormColumns, PreprocessingConfig } from './types';

interface ColumnConfigurationProps {
  formColumn: FormColumns['feature'][0];
  addTransformer: (
    transform: PreprocessingConfig,
    transformerGroup?: 'transforms' | 'featurizers'
  ) => void;
}

const ColumnPreprocessingPipelineInput = ({
  formColumn,
}: ColumnConfigurationProps) => {
  const [openTransformModal, setOpenTransformModal] = useState(false);
  const [openFeaturizerModal, setOpenFeaturizerModal] = useState(false);

  const { transforms, featurizers } = formColumn;

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
      {featurizers.map((transform) => (
        <TransformConstructorForm
          key={transform.name}
          transform={transform}
        />
      ))}
      {!DataTypeGuard.isNumericalOrQuantity(formColumn.col.dataType) &&
        featurizers.length === 0 && (
          <>
            <Text sx={{ width: '100%' }}>Featurizers:</Text>
            <NoData>
              <Link
                sx={{
                  textTransform: 'none',
                  cursor: 'pointer',
                  textDecoration: 'none',
                }}
                onClick={() => setOpenFeaturizerModal(true)}
              >
                Click here to add a featurizer
              </Link>
            </NoData>
          </>
        )}

      <Text sx={{ width: '100%' }}>Transforms:</Text>
      {transforms.length > 0 && (
        <CustomAccordion
          title="Transforms"
          sx={{
            minWidth: '70%',
          }}
        >
          <>
            {transforms.map((transform) => (
              <PreprocessingStepSelect value={transform} onChange={console.log}/>
            ))}

            <Button
              variant="outlined"
              color="primary"
              sx={{ padding: '1rem', fontSize: '20px' }}
              onClick={() => setOpenTransformModal(true)}
            >
              Add
            </Button>
          </>
        </CustomAccordion>
      )}
      {transforms.length === 0 && (
        <NoData>
          <Link
            sx={{
              textTransform: 'none',
              cursor: 'pointer',
              textDecoration: 'none',
            }}
            onClick={() => setOpenTransformModal(true)}
          >
            Click here to add a trasformer.
          </Link>
        </NoData>
      )}
    </Box>
  );
};

export default ColumnPreprocessingPipelineInput;

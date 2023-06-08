import { TargetConfig } from '@app/rtk/generated/models';
import { ModelOutputValue } from '@app/types/domain/models';
import { Text } from '@components/molecules/Text';
import ModelPrediction from '@features/models/components/ModelVersionInferenceView/ModelPrediction';
import { Box } from '@mui/material';

export const InferenceOutput = ({
  outputValues,
  targetColumns,
}: {
  outputValues: ModelOutputValue;
  targetColumns: TargetConfig[];
}) => (
  <>
    <Text fontWeight="bold" marginBottom={'0.5rem'} marginTop={'2rem'} key="1">
      Output:
    </Text>
    <Box
      key="2"
      sx={{
        display: 'flex',
        flexWrap: 'wrap',
        flexDirection: 'row',
        gap: '5px',
        ml: '8px',
        justifyContent: 'space-around',
      }}
    >
      {Object.keys(outputValues).map((key) => {
        const column = targetColumns.find((column) => column.name === key);
        if (!column) return null;

        const type =
          column.columnType == 'regression' ? 'numerical' : 'categorical';

        return (
          <Box
            key={column.name}
            sx={{
              mb: '1rem',
              mt: '1rem',
              border: '1px solid rgba(0, 0, 0, 0.12)',
              padding: '1rem',
              borderRadius: '4px',
              maxWidth: '360px',
              width: '100%',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'flex-start',
              alignItems: 'flex-start',
              height: '140px',
            }}
          >
            <ModelPrediction
              column={column.name}
              unit={'unit' in column.dataType ? column.dataType.unit : ''}
              type={type}
              // @ts-ignore
              value={outputValues[column.name]}
            />
          </Box>
        );
      })}
    </Box>
  </>
);

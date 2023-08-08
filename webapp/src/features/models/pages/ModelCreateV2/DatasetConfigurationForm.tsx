import { ModelCreate } from '@app/rtk/generated/models';
import { Section } from '@components/molecules/Section';
import DataPreprocessingInput from '@components/organisms/ModelBuilder/DataPreprocessingInput';
import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';
import { Box } from '@mui/material';
import { Controller, useFormContext } from 'react-hook-form';

export type GenericTransform = {
  name: string;
  constructorArgs: Record<string, any>;
  fowardArgs: Record<string, string | string[]>;
  type: string;
};

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

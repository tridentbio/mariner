import { ModelCreate } from '@app/rtk/generated/models';
import { Section } from '@components/molecules/Section';
import { Text } from '@components/molecules/Text';
import DataPreprocessingInput from '@components/organisms/ModelBuilder/DataPreprocessingInput';
import SklearnModelInput from '@components/organisms/ModelBuilder/SklearnModelInput';
import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';
import { Box, Step, StepContent, StepLabel, Stepper } from '@mui/material';
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
            <Stepper orientation="vertical">
              <Step active>
                <StepContent>
                  <StepLabel>
                    <Text variant="subtitle1">Feature columns</Text>
                  </StepLabel>
                  <DataPreprocessingInput
                    value={field.value.featureColumns as SimpleColumnConfig[]}
                    type="featureColumns"
                  />
                </StepContent>
              </Step>
              <Step active>
                <StepContent>
                  <StepLabel>
                    <Text variant="subtitle1">Target columns</Text>
                  </StepLabel>
                  <DataPreprocessingInput
                    value={field.value.targetColumns as SimpleColumnConfig[]}
                    type="targetColumns"
                  />
                </StepContent>
              </Step>
              {/*  <Step active>
              <StepContent>
                <StepLabel><Text variant="subtitle1">Model</Text></StepLabel>
                <SklearnModelInput />
              </StepContent>
            </Step> */}
            </Stepper>
          )}
        />
      </Section>
    </Box>
  );
};

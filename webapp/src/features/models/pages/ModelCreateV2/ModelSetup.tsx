import { ModelCreate } from '@app/rtk/generated/models';
import { Section } from '@components/molecules/Section';
import { Control, Controller } from 'react-hook-form';
import FeatureAndTargets from './FeatureAndTargets';
import { MenuItem, Select } from '@mui/material';

interface ModelSetupProps {
  control: Control<ModelCreate>;
}

export const ModelSetup = ({ control }: ModelSetupProps) => {
  return (
    <>
      <Section title="Features and Target">
        <FeatureAndTargets control={control} />
      </Section>
      <Section title="Framework">
        <Controller
          control={control}
          name="config.framework"
          rules={{
            required: {
              value: true,
              message: 'The framework is required',
            },
          }}
          render={({ field, fieldState }) => (
            <Select
              sx={{ width: '100%' }}
              {...(() => {
                console.log({ field });
                return field;
              })()}
            >
              <MenuItem value="torch">Torch</MenuItem>
              <MenuItem value="sklearn">Sklearn</MenuItem>
            </Select>
          )}
        />
      </Section>
    </>
  );
};
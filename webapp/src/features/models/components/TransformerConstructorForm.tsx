import { Control } from 'react-hook-form';
import { Transformer } from '../pages/ModelCreateV2/DatasetConfigurationForm';
import { ModelCreate } from '@app/rtk/generated/models';

type TransformerConstructorFormProps = {
  control: Control<ModelCreate>;
  transformer: Transformer;
};

export const TransformerConstructorForm = ({
  control,
  transformer,
}: TransformerConstructorFormProps) => {
  return <h3>{`${transformer.name}`}</h3>;
};

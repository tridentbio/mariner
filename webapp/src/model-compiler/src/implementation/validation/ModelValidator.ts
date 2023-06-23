import { TorchModelSpec } from '@app/rtk/generated/models';
import TransversalInfo from './TransversalInfo';

interface ModelValidator {
  validate(modelSchema: TorchModelSpec): TransversalInfo;
}

export default ModelValidator;

import { ModelSchema } from '../../interfaces/model-editor';
import TransversalInfo from './TransversalInfo';

interface ModelValidator {
  validate(modelSchema: ModelSchema): TransversalInfo;
}

export default ModelValidator;

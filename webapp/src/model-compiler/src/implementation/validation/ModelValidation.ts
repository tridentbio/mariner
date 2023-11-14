import {
  ComponentType,
  ModelSchema,
  NodeType,
} from '../../interfaces/torch-model-editor';
import { iterateTopologically } from '../../utils';
import Acceptor from './Acceptor';
import ModelValidator from './ModelValidator';
import TransversalInfo from './TransversalInfo';
import CategoricalRecommenderVisitor from './visitors/CategoricalRecommenderVisitor';
import ComponentVisitor from './visitors/ComponentVisitor';
import ConcatValidatorVisitor from './visitors/ConcatValidatorVisitor';
import GCNValidatorVisitor from './visitors/GCNValidatorVisitor';
import GlobalPoolingValidatorVisitor from './visitors/GlobalPoolingValidatorVisitor';
import LinearLinearWarningVisitor from './visitors/LinearLinearWarningVisitor';
import LinearValidatorVisitor from './visitors/LinearValidatorVisitor';
import MolFeaturizerValidatorVisitor from './visitors/MolFeaturizerValidatorVisitor';
import ShapeAndDataTypeVisitor from './visitors/ShapeAndDataTypeVisitor';
import SmilesRecommenderVisitor from './visitors/SmilesRecommenderVisitor';
import SoftmaxValidatorVisitor from './visitors/SoftmaxValidatorVisitor';

class ModelValidation extends Acceptor implements ModelValidator {
  validate = (modelSchema: ModelSchema): TransversalInfo => {
    const info = new TransversalInfo(modelSchema);
    const forwardVisitors = this.getVisitors();
    const backwardVisitors = this.getVisitors();

    this.iterateTopologicallyForward(modelSchema, (node, type) => {
      forwardVisitors.forEach((visitor) => {
        this.accept(visitor, { component: node, info, type, backward: false });
      });
    });

    this.iterateTopologicallyBackward(modelSchema, (node, type) => {
      info.updateInfoEdgesMap(node);
      backwardVisitors.forEach((visitor) => {
        this.accept(visitor, { component: node, info, type, backward: true });
      });
    });

    return info;
  };

  private getVisitors = (): ComponentVisitor[] => {
    return [
      new ShapeAndDataTypeVisitor(),
      new LinearValidatorVisitor(),
      new MolFeaturizerValidatorVisitor(),
      new GCNValidatorVisitor(),
      new SoftmaxValidatorVisitor(),
      new ConcatValidatorVisitor(),
      new LinearLinearWarningVisitor(),
      new CategoricalRecommenderVisitor(),
      new SmilesRecommenderVisitor(),
      new GlobalPoolingValidatorVisitor(),
    ];
  };

  private iterateTopologicallyForward = (
    schema: ModelSchema,
    fn: (node: NodeType, type: ComponentType) => void
  ) => {
    iterateTopologically(schema, fn, false);
  };

  private iterateTopologicallyBackward = (
    schema: ModelSchema,
    fn: (node: NodeType, type: ComponentType) => void
  ) => {
    iterateTopologically(schema, fn, true);
  };
}

export default ModelValidation;

import { iterateTopologically, unwrapDollar } from '../../utils';
import {
  ComponentType,
  ModelSchema,
  NodeType,
} from '../../interfaces/model-editor';
import Acceptor from './Acceptor';
import ModelValidator from './ModelValidator';
import TransversalInfo from './TransversalInfo';
import CategoricalRecommenderVisitor from './visitors/CategoricalRecommenderVisitor';
import ComponentVisitor from './visitors/ComponentVisitor';
import ConcatValidatorVisitor from './visitors/ConcatValidatorVisitor';
import GCNValidatorVisitor from './visitors/GCNValidatorVisitor';
import LinearLinearWarningVisitor from './visitors/LinearLinearWarningVisitor';
import LinearValidatorVisitor from './visitors/LinearValidatorVisitor';
import MolFeaturizerValidatorVisitor from './visitors/MolFeaturizerValidatorVisitor';
import ShapeAndDataTypeVisitor from './visitors/ShapeAndDataTypeVisitor';
import SmilesRecommenderVisitor from './visitors/SmilesRecommenderVisitor';
import SoftmaxValidatorVisitor from './visitors/SoftmaxValidatorVisitor';
import { isArray } from 'util';

class ModelValidation extends Acceptor implements ModelValidator {
  validate = (modelSchema: ModelSchema): TransversalInfo => {
    const info = new TransversalInfo(modelSchema);
    const forwardVisitors = this.getVisitors();
    const backwardVisitors = this.getVisitors();

    this.iterateTopologicallyForward(modelSchema, (node, type) => {
      this.updateInfoEdgesMap(info, node);
      forwardVisitors.forEach((visitor) => {
        this.accept(visitor, { component: node, info, type, backward: false });
      });
    });

    this.iterateTopologicallyBackward(modelSchema, (node, type) => {
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

  private updateInfoEdgesMap = (
    info: TransversalInfo,
    node: NodeType
  ): void => {
    if (node && 'forwardArgs' in node) {
      Object.entries(node.forwardArgs as object).forEach(
        ([_forwardArg, nodeAndEdge]) => {
          // todo: fix so concat works
          if (isArray(nodeAndEdge)) return;
          const [sourceNodeName, ...nodeOutputs] = nodeAndEdge.split('.');
          const key1 = unwrapDollar(sourceNodeName);
          // If output is a tensor, then key is "". If output is a
          const key2 = nodeOutputs.join('.');
          if (!info.edgesMap[key1]) {
            info.edgesMap[key1] = {};
          }
          console.log(`${key1}.${key2} <- ${node.name}`);
          info.edgesMap[key1][key2] = node.name;
        }
      );
    }
  };
}

export default ModelValidation;

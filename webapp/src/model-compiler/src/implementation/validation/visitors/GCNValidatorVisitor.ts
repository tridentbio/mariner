import { GcnConv } from 'model-compiler/src/interfaces/model-editor';
import { unwrapDollar } from '../../../utils';
import EditComponentsCommand, {
  makeComponentEdit,
} from '../../commands/EditComponentsCommand';
import Suggestion from '../../Suggestion';
import ComponentVisitor from './ComponentVisitor';

class GCNValidatorVisitor extends ComponentVisitor {
  visitGCN: ComponentVisitor['visitGCN'] = (input) => {
    if (input.backward) return this.visitGCNBackward(input);
    else return this.visitGCNForward(input);
  };

  visitGCNForward: ComponentVisitor['visitGCN'] = ({ component, info }) => {
    if (!component.forwardArgs.x) {
      return;
    }
    const x = unwrapDollar(component.forwardArgs.x);
    const shape = info.getOutgoingShapeSimple(x);
    if (!shape) return;
    const lastDim = shape.at(-1);
    if (lastDim === undefined) return;
    if (!isNaN(lastDim) && lastDim !== component.constructorArgs.in_channels) {
      const data = makeComponentEdit({
        component: component as GcnConv & {
          type: 'torch_geometric.nn.GcnConv';
        },
        constructorArgs: {
          in_channels: lastDim,
        },
      });
      info.addSuggestion(
        Suggestion.makeFixableConstructorArgsError(
          [
            new EditComponentsCommand({
              schema: info.schema,
              data,
            }),
          ],
          { nodeId: component.name }
        )
      );
    }
  };

  visitGCNBackward: ComponentVisitor['visitGCN'] = ({ component, info }) => {
    // gets the outgoing edges information of this node
    const edgeMap = info.edgesMap[component.name];
    if (!edgeMap) {
      return;
    }
    Object.entries(edgeMap).forEach(([outputAttribute, targetNodeName]) => {
      const requiredShape = info.getRequiredShape(
        targetNodeName,
        outputAttribute
      );
      if (
        requiredShape &&
        component.constructorArgs.out_channels != requiredShape.at(-1)
      ) {
        const data = makeComponentEdit({
          component: component,
          constructorArgs: {
            out_channels: requiredShape.at(-1),
          },
        });
        info.addSuggestion(
          Suggestion.makeFixableConstructorArgsError(
            [
              new EditComponentsCommand({
                schema: info.schema,
                data,
              }),
            ],
            { nodeId: component.name }
          )
        );
      }
    });
  };
}

export default GCNValidatorVisitor;

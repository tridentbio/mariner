import { GcnConv } from '@model-compiler/src/interfaces/model-editor';
import { unwrapDollar } from '../../../utils';
import EditComponentsCommand, {
  makeComponentEdit,
} from '../../commands/EditComponentsCommand';
import Suggestion from '../../Suggestion';
import ComponentVisitor from './ComponentVisitor';

class GCNValidatorVisitor extends ComponentVisitor {
  visitGCN: ComponentVisitor['visitGCN'] = (comp, _type, info) => {
    if (!comp.forwardArgs.x) {
      return;
    }
    const x = unwrapDollar(comp.forwardArgs.x);
    const shape = info.getShapeSimple(x);
    if (!shape) return;
    const lastDim = shape.at(-1);
    if (lastDim === undefined) return;
    if (!isNaN(lastDim) && lastDim !== comp.constructorArgs.in_channels) {
      const data = makeComponentEdit({
        component: comp as GcnConv & {
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
          { nodeId: comp.name }
        )
      );
    }
  };
}

export default GCNValidatorVisitor;

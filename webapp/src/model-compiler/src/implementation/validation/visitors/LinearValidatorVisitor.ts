import { Linear } from '../../../interfaces/model-editor';
import { unwrapDollar } from '../../../utils';
import EditComponentsCommand, {
  makeComponentEdit,
} from '../../commands/EditComponentsCommand';
import Suggestion from '../../Suggestion';
import ComponentVisitor from './ComponentVisitor';

class LinearValidatorVisitor extends ComponentVisitor {
  visitLinear: ComponentVisitor['visitLinear'] = (input) => {
    if (input.backward) {
      return this.visitLinearBackward(input);
    } else {
      return this.visitLinearForward(input);
    }
  };

  visitLinearForward: ComponentVisitor['visitLinear'] = ({
    component,
    info,
  }) => {
    const nodeId = unwrapDollar(component.forwardArgs.input);
    const shape = info.getOutgoingShapeSimple(nodeId);
    if (!shape) return;
    if (!component || !component.type) return;
    if (component.constructorArgs.in_features !== shape.at(-1)) {
      const data = makeComponentEdit({
        component: component as Linear & { type: 'torch.nn.Linear' },
        constructorArgs: {
          in_features: shape.at(-1),
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

  visitLinearBackward: ComponentVisitor['visitLinear'] = ({
    component,
    info,
  }) => {};
}

export default LinearValidatorVisitor;

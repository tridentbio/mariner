import { ComponentType, Linear } from '../../../interfaces/model-editor';
import { unwrapDollar } from '../../../utils';
import EditComponentsCommand, {
  makeComponentEdit,
} from '../../commands/EditComponentsCommand';
import Suggestion from '../../Suggestion';
import TransversalInfo from '../TransversalInfo';
import ComponentVisitor from './ComponentVisitor';

class LinearValidatorVisitor extends ComponentVisitor {
  visitLinear: (
    component: Linear,
    type: ComponentType,
    info: TransversalInfo
  ) => void = (component, _type, info) => {
    const nodeId = unwrapDollar(component.forwardArgs.input);
    const shape = info.getShapeSimple(nodeId);
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
}

export default LinearValidatorVisitor;

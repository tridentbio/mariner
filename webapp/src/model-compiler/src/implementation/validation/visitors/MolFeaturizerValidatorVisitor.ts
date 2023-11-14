import EditComponentsCommand, {
  makeComponentEdit,
} from '../../commands/EditComponentsCommand';
import { getDependenciesNames } from '../../modelSchemaQuery';
import Suggestion from '../../Suggestion';
import ComponentVisitor from './ComponentVisitor';

export default class MolFeaturizerValidatorVisitor extends ComponentVisitor {
  visitMolFeaturizer: ComponentVisitor['visitMolFeaturizer'] = (input) => {
    if (input.backward) {
      return this.visitMolFeaturizerBackward(input);
    } else {
      return this.visitMolFeaturizerForward(input);
    }
  };
  visitMolFeaturizerForward: ComponentVisitor['visitMolFeaturizer'] = ({
    component,
    info,
  }) => {
    // @ts-ignore
    const [mol] = getDependenciesNames(component);
    if (!mol) return;
    if (!component) return;
    if (info.getDataTypeSimple(mol)?.domainKind !== 'smiles') {
      const data = makeComponentEdit({
        // @ts-ignore
        component,
        removeConnection: {
          targetNodeForwardArg: 'mol',
        },
      });
      info.addSuggestion(
        Suggestion.makeFixableForwardArgsError(
          [
            new EditComponentsCommand({
              data,
            }),
          ],
          { edgeId: `${component.name}-${mol}` },
          "Can't connect non smiles input to MoleculeFeaturizer"
        )
      );
    }
  };

  visitMolFeaturizerBackward: ComponentVisitor['visitMolFeaturizer'] = (
    _input
  ) => {};
}

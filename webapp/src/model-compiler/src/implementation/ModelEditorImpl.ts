import {
  ComponentType,
  ModelSchema,
  NodePositionTypesMap,
} from '../interfaces/model-editor';
import AddComponentCommand, {
  AddCompArgs,
} from './commands/AddComponentCommand';
import ApplySuggestionsCommand, {
  ApplySuggestionsCommandArgs,
} from './commands/ApplySuggestionsCommand';
import DeleteComponentCommand, {
  DeleteCommandArgs,
} from './commands/DeleteComponentsCommand';
import EditComponentsCommand, {
  EditComponentsCommandArgs,
} from './commands/EditComponentsCommand';
import GetSuggestionsCommand, {
  GetSuggestionCommandArgs,
} from './commands/GetSuggestionCommand';
import TransversalInfo from './validation/TransversalInfo';

/**
 * Model schema transformations to support NN building
 */
class ModelEditorImpl {
  /**
   * Adds a component (layer or featurizer) into a model schema
   *
   * @template T - type of component, 'layer' or 'featurizer'
   * @param {AddCompArgs<T>} args - arguments of the AddComponentCommand
   * @returns {ModelSchema}
   */
  addComponent = <T extends ComponentType>(
    args: AddCompArgs<T>
  ): ModelSchema => {
    const command = new AddComponentCommand(args);
    return command.execute();
  };

  /**
   * Changes a existing component forwardArgs or constructorArgs
   *
   * @template T - type of component, 'layer' or 'featurizer'
   * @param {EditComponentsCommandArgs<T>} args - arguments of the EditComponentsCommand
   * @returns {ModelSchema}
   */
  editComponent = (args: EditComponentsCommandArgs): ModelSchema => {
    const command = new EditComponentsCommand(args);
    return command.execute();
  };

  /**
   * Deletes a component and it'sce dges
   *
   * @param {DeleteCommandArgs} args - arguments of the DeleteComponentCommand
   * @returns {ModelSchema}
   */
  deleteComponents = (args: DeleteCommandArgs): ModelSchema => {
    const command = new DeleteComponentCommand(args);
    return command.execute();
  };

  /**
   * Gets information about the model, includding suggestions, shapes and data types.
   *
   * @param {GetSuggestionCommandArgs} args - arguments of the GetSuggestionsCommand
   * @returns {TransversalInfo}
   */
  getSuggestions = (args: GetSuggestionCommandArgs): TransversalInfo => {
    const command = new GetSuggestionsCommand(args);
    return command.execute();
  };

  /**
   * Applies fixes (represented as commands) of suggestion's list
   *
   * @param {ApplySuggestionsCommandArgs} args - arguments of the ApplySuggestionsCommand
   * @returns {{schema: ModelSchema, updatedNodePositions: NodePositionTypesMap}}
   */
  applySuggestions = (
    args: ApplySuggestionsCommandArgs
  ): { schema: ModelSchema; updatedNodePositions: NodePositionTypesMap } => {
    const command = new ApplySuggestionsCommand(args);
    return command.execute();
  };
}

export default ModelEditorImpl;

import {
  ModelSchema,
  NodePositionTypesMap,
  NodeType,
} from '../../interfaces/torch-model-editor';
import Suggestion from '../Suggestion';
import AddComponentCommand from './AddComponentCommand';
import Command from './Command';
import EditComponentsCommand from './EditComponentsCommand';

export type ApplySuggestionsCommandArgs = {
  schema: ModelSchema;
  suggestions: Suggestion[];
};

class ApplySuggestionsCommand extends Command<
  ApplySuggestionsCommandArgs,
  { schema: ModelSchema; updatedNodePositions: NodePositionTypesMap }
> {
  constructor(args: ApplySuggestionsCommandArgs) {
    super(args);
  }
  execute = () => {
    let currentSchema = { ...this.args.schema };

    const updatedNodePositions: NodePositionTypesMap = {};

    this.args.suggestions.forEach((suggestion) => {
      suggestion.commands.forEach((command) => {
        if (
          command instanceof EditComponentsCommand ||
          command instanceof AddComponentCommand
        ) {
          if ('position' in command.args && command.args.position) {
            let nodeName: string;

            if (command instanceof EditComponentsCommand) {
              nodeName =
                typeof command.args.data == 'function'
                  ? command.args.data(currentSchema).name
                  : command.args.data.name;
            } else {
              nodeName = command.args.data.name;
            }

            updatedNodePositions[nodeName] = command.args.position;
          }
        }
        currentSchema = command.execute(currentSchema);
      });
    });

    return { schema: currentSchema, updatedNodePositions };
  };
}

export default ApplySuggestionsCommand;

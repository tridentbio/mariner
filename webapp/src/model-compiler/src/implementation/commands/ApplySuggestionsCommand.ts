import {
  ModelSchema,
  NodePositionTypesMap,
} from '../../interfaces/model-editor';
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
          if ('schema' in command.args) {
            command.args.schema = currentSchema;
          }
          if ('position' in command.args && command.args.position)
            updatedNodePositions[command.args.data.name] =
              command.args.position;
        }
        currentSchema = command.execute();
      });
    });

    return { schema: currentSchema, updatedNodePositions };
  };
}

export default ApplySuggestionsCommand;

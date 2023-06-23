import { ModelSchema } from '../../interfaces/model-editor';
import Suggestion from '../Suggestion';
import Command from './Command';
import EditComponentsCommand from './EditComponentsCommand';

export type ApplySuggestionsCommandArgs = {
  schema: ModelSchema;
  suggestions: Suggestion[];
};

class ApplySuggestionsCommand extends Command<
  ApplySuggestionsCommandArgs,
  ModelSchema
> {
  constructor(args: ApplySuggestionsCommandArgs) {
    super(args);
  }
  execute = (): ModelSchema => {
    let currentSchema = { ...this.args.schema };
    this.args.suggestions.forEach((suggestion) => {
      suggestion.commands.forEach((command) => {
        if (command instanceof EditComponentsCommand) {
          if ('schema' in command.args) {
            command.args.schema = currentSchema;
          }
        }
        currentSchema = command.execute();
      });
    });
    return currentSchema;
  };
}

export default ApplySuggestionsCommand;

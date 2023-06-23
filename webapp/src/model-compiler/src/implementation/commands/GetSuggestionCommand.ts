import { ModelSchema } from '../../interfaces/model-editor';
import ModelValidation from '../validation/ModelValidation';
import ModelValidator from '../validation/ModelValidator';
import TransversalInfo from '../validation/TransversalInfo';
import Command from './Command';

export type GetSuggestionCommandArgs = {
  schema: ModelSchema;
};

class GetSuggestionsCommand extends Command<
  GetSuggestionCommandArgs,
  TransversalInfo
> {
  private validator: ModelValidator = new ModelValidation();
  constructor(args: GetSuggestionCommandArgs) {
    super(args);
  }
  execute = (): TransversalInfo => {
    return this.validator.validate(this.args.schema);
  };
}

export default GetSuggestionsCommand;

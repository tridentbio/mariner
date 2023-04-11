import { ModelSchema } from '../interfaces/model-editor';
import Command from './commands/Command';
import EditComponentsCommand from './commands/EditComponentsCommand';
import SchemaContext from './SchemaContext';

type Severity = 'ERROR' | 'WARNING' | 'IMPROV';
class Suggestion {
  readonly severity: Severity;
  readonly message: string;
  readonly commands: Command<unknown, ModelSchema>[];
  readonly context: SchemaContext;

  constructor(
    severity: Severity,
    message: string,
    commands: Command<any, ModelSchema>[],
    context: SchemaContext
  ) {
    this.severity = severity;
    this.commands = commands;
    this.message = message;
    this.context = context;
  }

  static makeFixableConstructorArgsError = (
    commands: Command<unknown, ModelSchema>[],
    context: SchemaContext,
    message?: string
  ) => {
    return new Suggestion(
      'ERROR',
      message || 'The constructor arguments of this layer result in an error',
      commands,
      context
    );
  };

  static makeFixableForwardArgsError = (
    commands: Command<unknown, ModelSchema>[],
    context: SchemaContext,
    message?: string
  ) => {
    return new Suggestion(
      'ERROR',
      message ||
        'The forward arguments of this layer or featurizer results in an error',
      commands,
      context
    );
  };

  static makeForwardArgsError = (context: SchemaContext, message?: string) =>
    new Suggestion(
      'ERROR',
      message || 'Got invalid forward arguments',
      [],
      context
    );

  getCurrentConstructorArgs = (name: string) => {
    for (const command of this.commands) {
      if (command instanceof EditComponentsCommand && command.args.schema) {
        let obj;
        obj = (command.args.schema.layers || []).find(
          (schemaObj: any) => schemaObj.name === name
        );
        obj =
          obj ||
          (command.args.schema.featurizers || []).find(
            (schemaObj: any) => schemaObj.name === name
          );
        // ignored because the constructorArgs property is not always in obj but ts doesn't know that
        // @ts-ignore
        if (obj && obj.constructorArgs) return obj.constructorArgs;
      }
    }
  };

  getConstructorArgsErrors = () => {
    return this.commands.reduce((acc, command) => {
      if (command instanceof EditComponentsCommand) {
        if (
          typeof command.args !== 'object' ||
          !('constructorArgs' in command.args.data)
        )
          return acc;

        const currentConstructorArgs = this.getCurrentConstructorArgs(
          command.args.data.name
        ) as typeof command.args.data.constructorArgs;

        const isWrong = currentConstructorArgs
          ? (value: any, constructorArgs: string) =>
              // @ts-ignore
              currentConstructorArgs[constructorArgs] !== value
          : () => true;

        const errors = Object.entries(
          command.args.data.constructorArgs || {}
        ).reduce((acc, [constructorArgs, value]) => {
          if (isWrong(value, constructorArgs))
            acc[constructorArgs] = `Should be ${value}`;
          return acc;
        }, {} as Record<string, string>);
        return { ...acc, ...errors };
      }
      return acc;
    }, {} as Record<string, string>);
  };
}

export default Suggestion;

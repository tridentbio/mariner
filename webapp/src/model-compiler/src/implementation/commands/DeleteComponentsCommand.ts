import {
  FeaturizersType,
  LayersType,
  ModelSchema,
} from '../../interfaces/model-editor';
import Command from './Command';

export type DeleteCommandArgs = {
  nodeId: string;
  readonly schema: ModelSchema;
};

class DeleteComponentCommand extends Command<DeleteCommandArgs, ModelSchema> {
  constructor(args: DeleteCommandArgs) {
    super(args);
  }
  private removeBrokenEdges = <T extends LayersType | FeaturizersType>(
    component: T
  ): T => {
    if ('forwardArgs' in component) {
      const forwardArgs = {};
      Object.keys(component.forwardArgs).forEach((key) => {
        const value = component.forwardArgs[
          key as keyof typeof component.forwardArgs
        ] as string | undefined;
        // don't include edge from deleted node
        if (value && value.includes(this.args.nodeId)) return;
        forwardArgs[key as keyof typeof component.forwardArgs] =
          component.forwardArgs[key as keyof typeof component.forwardArgs];
      });
      return {
        ...component,
        forwardArgs,
      };
    }
    return component;
  };
  execute = (): ModelSchema => {
    return {
      ...this.args.schema,
      layers: (this.args.schema.layers || [])
        .filter((layer) => layer.name !== this.args.nodeId)
        .map(this.removeBrokenEdges),
      featurizers: (this.args.schema.featurizers || [])
        .filter((featurizer) => featurizer.name !== this.args.nodeId)
        .map(this.removeBrokenEdges),
    };
  };
}

export default DeleteComponentCommand;

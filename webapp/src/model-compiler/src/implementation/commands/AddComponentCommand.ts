import {
  ComponentType,
  Embedding,
  FeaturizersType,
  LayersType,
  ModelSchema,
  NodePositionTypes,
} from '../../interfaces/model-editor';
import { wrapForwardArgs } from '../../utils';
import Command from './Command';

export type AddCompArgs<T extends ComponentType> = {
  schema: ModelSchema;
  readonly type: T;
  readonly data: T extends 'layer'
    ? LayersType
    : T extends 'featurizer'
    ? FeaturizersType
    : never;
  position?: NodePositionTypes;
};

class AddComponentCommand<T extends ComponentType> extends Command<
  AddCompArgs<T>,
  ModelSchema
> {
  constructor(args: AddCompArgs<T>) {
    super(args);
  }
  execute = (): ModelSchema => {
    this.args.data.forwardArgs = wrapForwardArgs(this.args.data.forwardArgs);
    if (this.args.type === 'layer') {
      const layers = [...(this.args.schema.spec.layers || [])];
      // Hack to make up for deffective backend default construction arguments
      // for the Embedding layer
      if (this.args.data.type === 'torch.nn.Embedding') {
        const data = { ...this.args.data };
        (data as Embedding).constructorArgs['max_norm'] = 1;
        (data as Embedding).constructorArgs['norm_type'] = 2;
        layers.push(data as LayersType);
      } else layers.push(this.args.data as LayersType);
      return {
        ...this.args.schema,
        spec: {
          layers,
        },
      };
    } else if (this.args.type === 'featurizer') {
      const featurizers = [...(this.args.schema.dataset.featurizers || [])];
      featurizers.push(this.args.data as FeaturizersType);
      return {
        ...this.args.schema,
        dataset: {
          ...this.args.schema.dataset,
          featurizers,
        },
      };
    } else {
      throw new Error('Not implemented');
    }
  };
}

export default AddComponentCommand;

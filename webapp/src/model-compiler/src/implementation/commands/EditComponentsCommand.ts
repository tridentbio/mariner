import { GetModelOptionsApiResponse } from 'app/rtk/generated/models';
import { ArrayElement } from 'utils';
import {
  Concat,
  FeaturizersType,
  LayersType,
  ModelSchema,
  NodeType,
  Output,
} from '../../interfaces/model-editor';
import Command from './Command';

const isForwardArgList = (
  component: NodeType['type'],
  forwardArg: string,
  options: Record<string, ArrayElement<GetModelOptionsApiResponse>> = {}
): boolean => {
  if (!component || component === 'output') return false;
  if (component in options) {
    const option = options[component];
    const argsSummary = option.component.forwardArgsSummary;
    if (!argsSummary) return false;
    if (!(forwardArg in argsSummary))
      throw `${forwardArg} not found in forwardArgsSummary`;
    // @ts-ignore
    const forwardArgType = argsSummary[forwardArg];
    return forwardArgType.toLowerCase().includes('list');
  }
  throw `${component} not found in options dict! This is a temporary error.`;
  return false;
};

export type EditComponentsCommandArgs = {
  schema: ModelSchema;
  data: NodeType;
};

class EditComponentsCommand extends Command<
  EditComponentsCommandArgs,
  ModelSchema
> {
  constructor(args: EditComponentsCommandArgs) {
    super(args);
  }
  execute = (): ModelSchema => {
    // Handle target column connection
    const targetColumnIndex =
      this.args.data.type === 'output'
        ? this.args.schema.dataset.targetColumns.findIndex(
            (targetColumn) => this.args.data.name === targetColumn.name
          )
        : -1;
    if (targetColumnIndex !== -1) {
      const targetColumns = [...this.args.schema.dataset.targetColumns];
      targetColumns[targetColumnIndex] = this.args.data as Output;
      return {
        ...this.args.schema,
        dataset: {
          ...this.args.schema.dataset,
          targetColumns: targetColumns,
        },
      };
    }
    const layers = this.args.schema.spec.layers || [];
    const updatedLayers = layers.map((layer) => {
      if (layer.name !== this.args.data.name) return layer;
      return this.args.data as LayersType;
    });
    const featurizers = this.args.schema.dataset.featurizers || [];
    const updatedFeaturizers = featurizers.map((featurizer) => {
      if (featurizer.name !== this.args.data.name) return featurizer;
      else return this.args.data as FeaturizersType;
    });
    return {
      ...this.args.schema,
      spec: { layers: updatedLayers },
      dataset: {
        ...this.args.schema.dataset,
        featurizers: updatedFeaturizers,
      },
    };
  };
}

type Connection = {
  sourceComponentName: string;
  sourceComponentOutput?: string;
  targetNodeForwardArg: string;
};
type RemoveConnection = {
  targetNodeForwardArg: string;
  elementValue?: string;
};

export const makeComponentEdit = <T extends NodeType>(params: {
  component: T;
  options?: Record<string, ArrayElement<GetModelOptionsApiResponse>>;
  constructorArgs?: T extends { constructorArgs: infer C }
    ? Partial<C>
    : undefined;
  forwardArgs?: Connection;
  removeConnection?: RemoveConnection;
  lossFn?: string;
  columnType?: string;
}): T => {
  const { component, options, constructorArgs, forwardArgs, removeConnection } =
    params;
  const newComponent = { ...component };
  if (newComponent.type && ['input'].includes(newComponent.type))
    return newComponent;
  if ('constructorArgs' in newComponent && constructorArgs) {
    newComponent.constructorArgs = {
      ...newComponent.constructorArgs,
      ...constructorArgs,
    };
  }

  ['lossFn', 'columnType'].forEach((key) => {
    if (key in params) {
      // @ts-ignore
      newComponent[key] = params[key];
    }
  });

  if ('forwardArgs' in newComponent && forwardArgs) {
    const { targetNodeForwardArg, sourceComponentOutput, sourceComponentName } =
      forwardArgs;
    const edgeName = `$${[sourceComponentName, sourceComponentOutput]
      .filter((el) => !!el)
      .join('.')}`;
    if (isForwardArgList(newComponent.type, targetNodeForwardArg, options)) {
      // @ts-ignore
      newComponent.forwardArgs[targetNodeForwardArg] = [
        // @ts-ignore
        ...(newComponent.forwardArgs[targetNodeForwardArg] || []),
        edgeName,
      ];
    } else if (component.type === 'output') {
      // @ts-ignore
      newComponent.forwardArgs[targetNodeForwardArg] = edgeName;
      // @ts-ignore
      newComponent.outModule = edgeName.replace('$', '');
    } else {
      // @ts-ignore
      newComponent.forwardArgs[targetNodeForwardArg] = edgeName;
    }
  }

  if ('forwardArgs' in newComponent && removeConnection) {
    const { targetNodeForwardArg, elementValue } = removeConnection;
    if (
      elementValue &&
      isForwardArgList(newComponent.type, targetNodeForwardArg)
    ) {
      // @ts-ignore
      const fArgs = newComponent.forwardArgs as Concat['forwardArgs'];
      fArgs[targetNodeForwardArg as 'xs'] = (
        fArgs[targetNodeForwardArg as 'xs'] || []
      ).filter((value) => value !== `$${elementValue}`);
    } else if (component.type === 'output') {
      // @ts-ignore
      newComponent.forwardArgs[targetNodeForwardArg] = '';
      // @ts-ignore
      newComponent.outModule = undefined;
    } else {
      // @ts-ignore
      newComponent.forwardArgs[targetNodeForwardArg] = '';
    }
  }
  return newComponent;
};

export default EditComponentsCommand;

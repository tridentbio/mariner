import { ModelSchema } from '@app/rtk/generated/models';
import { NodeType } from '@model-compiler/src/interfaces/model-editor';
import { unwrapDollar } from '@model-compiler/src/utils';
import { flatten } from '@utils';
import { getNodes } from '../modelSchemaQuery';
import Command from './Command';

type SourceEndpoint = {
  type: 'source';
  component: NodeType;
  // The name of the output handle if there are multiple ones (like with MoleculeFeaturizer)
  outputAttributeName?: string;
};

type TargetEndpoint = {
  type: 'target';
  component: NodeType;
  forwardArg: string;
};

type Endpoint = SourceEndpoint | TargetEndpoint;

type Position = { x: number; y: number };

const isSourceEndpoint = (a: any): a is SourceEndpoint =>
  !!a && 'type' in a && a.type === 'source';
const isTargetEndpoint = (a: any): a is TargetEndpoint =>
  !!a && 'type' in a && a.type === 'target';

export const getConnectableArguments = (
  endpoint: Endpoint,
  schema: ModelSchema,
  _positionsMap?: Record<string, Position>
): Endpoint[] => {
  const nodes = getNodes(schema);
  if (isSourceEndpoint(endpoint)) {
    const targetEndpoints: TargetEndpoint[] = [];
    nodes.forEach((node) => {
      if (
        node.type === 'input' ||
        node.type === 'output' ||
        node.name === endpoint.component.name
      )
        return;
      Object.entries(node.forwardArgs || {}).forEach(([key, value]) => {
        if (!!value) return;
        targetEndpoints.push({
          type: 'target',
          component: node,
          forwardArg: key,
        });
      });
    });
    return targetEndpoints;
  } else if (isTargetEndpoint(endpoint)) {
    const sourceEndpoints: SourceEndpoint[] = [];
    const sourceEndpointUnfilled: Record<string, boolean> = {};
    flatten(
      nodes.map((node) => {
        if (node.type === 'input') {
          return [node.name];
        } else if (node.type === 'output') {
          return [];
        } else if (
          node.type === 'model_builder.featurizers.MoleculeFeaturizer'
        ) {
          return ['x', 'edge_index', 'edge_attr', 'batch'].map(
            (output) => `${node.name}.${output}`
          );
        } else {
          return [node.name];
        }
      })
    ).forEach((str) => {
      sourceEndpointUnfilled[str] = true;
    });
    nodes.forEach((node) => {
      if (node.type === 'input' || node.type === 'output') return;
      Object.values(node.forwardArgs).forEach((value: string) => {
        if (!value) return;
        const keyy = unwrapDollar(value);
        sourceEndpointUnfilled[keyy] = false;
      });
    });
    const componentsByNodeName: Record<string, NodeType> = {};
    nodes.forEach((node) => {
      componentsByNodeName[node.name] = node;
    });
    Object.entries(sourceEndpointUnfilled).forEach(([key, unfilled]) => {
      if (unfilled) {
        const [nodeName, attribute] = key.split('.');
        const outputAttributeName = attribute || '';
        sourceEndpoints.push({
          outputAttributeName,
          type: 'source',
          component: componentsByNodeName[nodeName],
        });
      }
    });
    return sourceEndpoints;
  }
  return [];
};

export const getConnectableArgumentsByNode = <T extends Endpoint>(
  endpoint: T,
  schema: ModelSchema,
  positionsMap?: Record<string, Position>
): T extends SourceEndpoint
  ? Record<string, TargetEndpoint[]>
  : Record<string, SourceEndpoint[]> => {
  const endpoints = getConnectableArguments(endpoint, schema, positionsMap);
  //@ts-ignore
  return endpoints.reduce((acc, cur) => {
    if (endpoint.type === 'source') {
      if (cur.type === 'source') return acc;
      if (!(cur.component.name in acc)) {
        acc[cur.component.name] = [cur];
        return acc;
      }
      (acc as Record<string, TargetEndpoint[]>)[cur.component.name].push(cur);
      return acc;
    } else {
      // TODO
    }
  }, {} as T extends SourceEndpoint ? Record<string, TargetEndpoint[]> : Record<string, SourceEndpoint[]>);
};

export type GetConnectionsCommandArgs<T extends Endpoint> = {
  schema: ModelSchema;
  endpoint: T;
  positionsMap?: Record<string, Position>;
};

export default class GetPossibleConnectionsCommand<
  T extends Endpoint
> extends Command<
  GetConnectionsCommandArgs<T>,
  T extends SourceEndpoint
    ? Record<string, TargetEndpoint[]>
    : Record<string, SourceEndpoint[]>
> {
  constructor(args: GetConnectionsCommandArgs<T>) {
    super(args);
  }

  execute = () => {
    if (isSourceEndpoint(this.args.endpoint)) {
      return getConnectableArgumentsByNode(
        this.args.endpoint,
        this.args.schema,
        this.args.positionsMap
      );
    } else if (isTargetEndpoint(this.args.endpoint)) {
      return getConnectableArgumentsByNode(
        this.args.endpoint,
        this.args.schema,
        this.args.positionsMap
      );
    }
    return {};
  };
}

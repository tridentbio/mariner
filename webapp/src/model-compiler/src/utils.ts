import { TorchModelSpec } from '@app/rtk/generated/models';
import {
  getDependencies,
  getDependents,
  getNodes,
} from './implementation/modelSchemaQuery';
import {
  ComponentType,
  ModelSchema,
  NodeType,
} from './interfaces/model-editor';

type ForwardArgs = {
  [key: string]: string | string[];
};

export const isArray = (t: any): t is unknown[] => {
  return t?.length !== undefined && !(typeof t === 'string');
};
export const wrapDollar = (str: string): string => {
  if (!str) return '';
  return str[0] === '$' ? str : `$${str}`;
};
export const unwrapDollar = (str: any): string => {
  if (!str || typeof str !== 'string') return '';
  if (str.startsWith('${') && str.endsWith('}'))
    return str.substring(2, str.length - 1);
  if (str.startsWith('$')) return str.substring(1);
  return str;
};

export const wrapForwardArgs = <T extends ForwardArgs>(forwardArgs: T): T => {
  let newForwardArgs = {} as {
    [K in keyof T]: T[K];
  };
  Object.keys(forwardArgs).forEach((key: keyof T) => {
    const value = forwardArgs[key];
    if (isArray(value)) {
      // @ts-ignore
      newForwardArgs[key] = value.map(wrapDollar);
    } else {
      // @ts-ignore
      newForwardArgs[key] = wrapDollar(value);
    }
  });
  return newForwardArgs;
};

export const arrayEquals = (arr1: number[], arr2: number[]) =>
  arr1.length == arr2.length &&
  arr1.reduce((acc, val, idx) => acc && val === arr2[idx], true);

export const normalizeNegativeIndex = (index: number, length: number) => {
  if (index < 0) return index + length;
  else return index;
};

export const iterateTopologically = (
  schema: ModelSchema,
  fn: (node: NodeType, type: ComponentType) => void,
  backward = false
) => {
  const nodes = topologicalSort(schema, backward);
  const typesOfNode: {
    [key: string]: ComponentType;
  } = {};
  if (schema.spec.layers)
    schema.spec.layers.forEach((layer) => {
      typesOfNode[layer.name] = 'layer' as const;
    });
  if (schema.dataset.featurizers)
    schema.dataset.featurizers.forEach((featurizer) => {
      typesOfNode[featurizer.name] = 'featurizer';
    });
  nodes.forEach((node) => {
    const type = typesOfNode[node.name] || node.type;
    fn(node, type);
  });
};

const topologicalSortHelper = (
  node: NodeType,
  explored: Set<string>,
  stack: NodeType[],
  schema: ModelSchema,
  backward = false
) => {
  explored.add(node.name);

  type DepCollector = (node: NodeType, schema: ModelSchema) => NodeType[];

  let collector: DepCollector;

  if (backward) collector = getDependents;
  else collector = getDependencies;
  collector(node, schema).forEach((n) => {
    if (!explored.has(n.name))
      topologicalSortHelper(n, explored, stack, schema);
  });
  stack.push(node);
};

const topologicalSort = (schema: ModelSchema, backward = false): NodeType[] => {
  if (backward) return topologicalSortBackward(schema);
  else return topologicalSortForward(schema);
};

const topologicalSortForward = (schema: ModelSchema): NodeType[] => {
  const stack: NodeType[] = [];
  const explored = new Set<string>();
  const nodes = getNodes(schema);
  nodes.forEach((node) => {
    if (!explored.has(node.name)) {
      topologicalSortHelper(node, explored, stack, schema);
    }
  });
  return stack;
};

const topologicalSortBackward = (schema: ModelSchema): NodeType[] => {
  const stack: NodeType[] = [];
  const explored = new Set<string>();
  const nodes = getNodes(schema);
  nodes.forEach((node) => {
    if (!explored.has(node.name)) {
      topologicalSortHelper(node, explored, stack, schema, true);
    }
  });
  return stack;
};

export const extendSpecWithTargetForwardArgs = (
  spec: TorchModelSpec
): ModelSchema => {
  return {
    ...spec,
    dataset: {
      ...spec.dataset,
      featurizers: spec.dataset.featurizers || [],
      targetColumns: spec.dataset.targetColumns.map((tc) => ({
        ...tc,
        forwardArgs: { '': '' },
      })),
    },
  };
};

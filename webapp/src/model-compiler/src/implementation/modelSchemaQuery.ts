import { ArrayElement, flatten } from 'utils';
import {
  DataType,
  ModelOptions,
  ModelSchema,
  NodeType,
  EPythonClasses,
} from '../interfaces/model-editor';
import { isArray, unwrapDollar } from '../utils';

const len = (t: { length?: number } | undefined) => t?.length || 0;
export const getNodes = (schema: ModelSchema): NodeType[] => {
  let layersAndFeats: NodeType[] = (schema.dataset.featurizers || []).concat(
    // @ts-ignore
    schema.spec.layers || []
  );

  layersAndFeats = layersAndFeats.concat(
    schema.dataset.targetColumns.map((targetColumn) => {
      return {
        type: 'output',
        name: targetColumn.name,
        dataType: targetColumn.dataType as DataType,
        columnType: targetColumn.columnType,
        lossFn: targetColumn.lossFn,
        outModule: targetColumn.outModule,
        forwardArgs: targetColumn.outModule
          ? { '': `$${targetColumn.outModule}` }
          : targetColumn.forwardArgs || { '': '' },
      };
    })
  );
  layersAndFeats = layersAndFeats.concat(
    schema.dataset.featureColumns.map((col) => ({
      type: 'input',
      name: col.name,
      dataType: col.dataType as DataType,
    }))
  );
  return layersAndFeats;
};

export const getNode = (schema: ModelSchema, nodeName: string) => {
  return getNodes(schema).find((node) => node.name === nodeName);
};

export const getComponent = (
  schema: ModelSchema,
  componentName: string
): NodeType => {
  const layer =
    schema.spec.layers &&
    schema.spec.layers.find((l) => l.name === componentName);
  if (layer) return layer;
  const featurizer =
    schema.dataset.featurizers &&
    schema.dataset.featurizers.find((f) => f.name === componentName);
  if (featurizer) return featurizer;
  const targetColumn = schema.dataset.targetColumns.find(
    (col) => col.name === componentName
  );
  if (targetColumn)
    return {
      type: 'output',
      name: targetColumn.name,
      dataType: targetColumn.dataType,
      forwardArgs: targetColumn.forwardArgs || { '': '' },
      outModule: targetColumn.outModule,
      columnType: targetColumn.columnType,
      lossFn: targetColumn.lossFn,
    };
  const input = schema.dataset.featureColumns.find(
    (col) => col.name === componentName
  );
  if (input)
    return {
      type: 'input',
      dataType: input.dataType,
      name: input.name,
    };
  throw 'component not found. This error is temporary';
};

export const getDependenciesNames = (node: NodeType): string[] => {
  const deps: string[] = [];
  if (node.type === 'input' || node.type === 'output') return [];
  Object.values(node.forwardArgs).forEach((value) => {
    if (isArray(value)) {
      value.forEach((val) => {
        const nodeId = unwrapDollar(val);
        // only interested in node order
        const [head] = nodeId.split('.');
        deps.push(head);
      });
    } else {
      const [head] = unwrapDollar(value).split('.');
      deps.push(head);
    }
  });
  return deps;
};
export const getDependencies = (
  node: NodeType,
  schema: ModelSchema
): NodeType[] => {
  const nodes = getNodes(schema);
  const nodesByName: { [key: string]: NodeType } = {};
  nodes.forEach((n) => {
    nodesByName[n.name] = n;
  });
  const depsNames = getDependenciesNames(node);
  const deps = depsNames
    .map((nodeId) => nodesByName[nodeId])
    .filter((name) => !!name);
  return deps;
};

type HandleData = {
  nodeId: string;
  handle: string;
  type: 'source' | 'target';
  isConnected: boolean;
  connectedTo?: {
    nodeId: string;
    handle: string;
    type: 'source' | 'target';
  };
  acceptMultiple?: boolean;
};

/**
 * Get's the target handle objects, an array of the forwardArgs state from
 * each of the components in schema
 *
 * @param {ModelSchema} schema - schema to get the target handles from
 * @returns {HandleData[]}
 */
export const getTargetHandles = (schema: ModelSchema): HandleData[] => {
  const nodes = getNodes(schema);
  const result: HandleData[] = [];
  nodes.forEach((node) => {
    if (node.type === 'input') return;
    else if (node.type === 'output')
      result.push({
        nodeId: node.name,
        handle: '',
        type: 'target',
        isConnected: false, // schema has no info about target col edge
      });
    else {
      Object.entries(node.forwardArgs).forEach(([handle, value]) => {
        const [head, ...tail] = unwrapDollar(value || '').split('.');
        const handleData: HandleData = {
          nodeId: node.name,
          handle,
          type: 'target',
          isConnected: !!head,
        };
        if (handleData.isConnected) {
          handleData.connectedTo = {
            nodeId: head,
            handle: tail.join('.'),
            type: 'source',
          };
        }
        result.push(handleData);
      });
    }
  });

  return result;
};

/**
 * Get's the source handle objects inspecting the source direction from the
 * layers defined in the forwardArgs attribute of the components of the schema
 *
 * @param {ModelSchema} schema
 * @param {ModelOptions} options - array with output type information of components
 * @returns {HandleData[]} array of source handles (or the data to complete a connection)
 */
export const getSourceHandles = (
  schema: ModelSchema,
  options: ModelOptions
): HandleData[] => {
  const optionsByType: Record<string, ArrayElement<ModelOptions>> = {};
  options.forEach((option) => {
    if (!option.component?.type) return;
    optionsByType[option.component.type] = option;
  });
  const nodes = getNodes(schema);
  const handlesDataByNode: Record<string, Omit<HandleData, 'key'>[]> = {};

  // Collect the handles data with unknown connected state
  nodes.forEach((node) => {
    if (!handlesDataByNode[node.name]) handlesDataByNode[node.name] = [];
    if (node.type === 'output') return;
    else if (node.type === 'input') {
      handlesDataByNode[node.name] = [
        {
          type: 'source',
          nodeId: node.name,
          handle: '',
          isConnected: false,
        },
      ];
    } else {
      if (!node.type) return;
      const option = optionsByType[node.type];
      if (!option || !('outputType' in option)) return;
      const outputType = option.outputType;
      if (outputType === EPythonClasses.TORCH_GEOMETRIC_DATA_REQUIRED) {
        handlesDataByNode[node.name] = [
          'x',
          'edge_index',
          'edge_attr',
          'batch',
        ].map((handle) => ({
          nodeId: node.name,
          type: 'source',
          handle,
          isConnected: false,
        }));
      } else {
        const handleData: HandleData = {
          type: 'source',
          nodeId: node.name,
          isConnected: false, // updated in later pass
          handle: '',
        };
        handlesDataByNode[node.name].push(handleData);
      }
    }
  });

  // Fill connected state of handles
  nodes.forEach((node) => {
    if (node.type === 'input' || node.type === 'output') return;
    Object.entries(node.forwardArgs).forEach(([key, value]) => {
      if (!value) return;
      const [head, ...tail] = unwrapDollar(value || '').split('.');
      if (!(head in handlesDataByNode))
        throw `Reference to unknown component ${head}`;
      const handle = tail.join('.');

      handlesDataByNode[head].forEach((handleData) => {
        if (handleData.handle === handle) {
          handleData.isConnected = true;
          handleData.connectedTo = {
            type: 'target',
            nodeId: node.name,
            handle: key,
          };
        }
      });
    });
  });

  return flatten(Object.values(handlesDataByNode));
};

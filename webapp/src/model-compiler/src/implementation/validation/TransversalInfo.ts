import {
  FleetaddpoolingForwardArgsReferences,
  FleetconcatForwardArgsReferences,
  FleetdnasequencefeaturizerForwardArgsReferences,
  FleetglobalpoolingForwardArgsReferences,
  FleetintegerfeaturizerForwardArgsReferences,
  FleetonehotForwardArgsReferences,
  FleetproteinsequencefeaturizerForwardArgsReferences,
  FleetrnasequencefeaturizerForwardArgsReferences,
  TorchembeddingForwardArgsReferences,
  TorchgeometricgcnconvForwardArgsReferences,
  TorchlinearForwardArgsReferences,
  TorchreluForwardArgsReferences,
  TorchsigmoidForwardArgsReferences,
  TorchtransformerencoderlayerForwardArgsReferences,
} from '@app/rtk/generated/models';
import {
  DataType,
  FeaturizersType,
  LayersType,
  ModelSchema,
  NodeType,
  Output,
} from '../../interfaces/torch-model-editor';
import Suggestion from '../Suggestion';
import { getDependents, getNodes } from '../modelSchemaQuery';
import { unwrapDollar } from '../../utils';

type NodeEdgeEntry<K extends string, T> = {
  type: K;
  edges: { [edgeId in keyof T]: string[] };
};

type NodeEdgeTypesMap = {
  'fleet.model_builder.layers.OneHot': FleetonehotForwardArgsReferences;
  'fleet.model_builder.layers.GlobalPooling': FleetglobalpoolingForwardArgsReferences;
  'fleet.model_builder.layers.Concat': FleetconcatForwardArgsReferences;
  'fleet.model_builder.layers.AddPooling': FleetaddpoolingForwardArgsReferences;
  'torch.nn.Linear': TorchlinearForwardArgsReferences;
  'torch.nn.Sigmoid': TorchsigmoidForwardArgsReferences;
  'torch.nn.ReLU': TorchreluForwardArgsReferences;
  'torch_geometric.nn.GCNConv': TorchgeometricgcnconvForwardArgsReferences;
  'torch.nn.Embedding': TorchembeddingForwardArgsReferences;
  'torch.nn.TransformerEncoderLayer': TorchtransformerencoderlayerForwardArgsReferences;
  'fleet.model_builder.featurizers.MoleculeFeaturizer': TorchtransformerencoderlayerForwardArgsReferences;
  'fleet.model_builder.featurizers.IntegerFeaturizer': FleetintegerfeaturizerForwardArgsReferences;
  'fleet.model_builder.featurizers.DNASequenceFeaturizer': FleetdnasequencefeaturizerForwardArgsReferences;
  'fleet.model_builder.featurizers.RNASequenceFeaturizer': FleetrnasequencefeaturizerForwardArgsReferences;
  'fleet.model_builder.featurizers.ProteinSequenceFeaturizer': FleetproteinsequencefeaturizerForwardArgsReferences;
  output: Output['forwardArgs'];
  default: NodeType['type'];
};

export type NodeEdgeTypes<T extends string = ''> =
  T extends keyof NodeEdgeTypesMap
    ? NodeEdgeEntry<T, NodeEdgeTypesMap[T]>
    : NodeEdgeEntry<NodeType['type'], { [key: string]: string[] }>;

type BiDimensionalDictionary<V> = {
  [key1: string]: {
    [key2: string]: V;
  };
};

const getFromMap = <T>(
  key1: string,
  key2: string,
  map: BiDimensionalDictionary<T>
) => {
  if (key1 in map && key2 in map[key1]) {
    return map[key1][key2];
  }
};

const setMap = <T>(
  key1: string,
  key2: string,
  value: T,
  map: BiDimensionalDictionary<T>
) => {
  if (!(key1 in map)) map[key1] = {};
  map[key1][key2] = value;
};

class TransversalInfo {
  suggestions: Suggestion[] = [];
  outgoingShapes: BiDimensionalDictionary<number[]> = {};
  requiredShapes: BiDimensionalDictionary<number[]> = {};
  dataTypes: BiDimensionalDictionary<DataType> = {};
  /** Used to answer queries like: "What nodes are connected to the node X following the edge Y?" */
  edgesMap: { [nodeId: string]: NodeEdgeTypes } = {};

  readonly schema: ModelSchema;
  readonly nodesByName: {
    [key: string]: LayersType | FeaturizersType;
  };

  constructor(schema: ModelSchema) {
    this.nodesByName = {};
    this.schema = schema;
    getNodes(schema).forEach((node) => {
      if (node.type === 'input' || node.type === 'output') return;
      this.nodesByName[node.name] = node;
    });
    schema.dataset.targetColumns.forEach((targetColumn) => {
      if (targetColumn.outModule && targetColumn.forwardArgs) {
        const edgeId = Object.keys(targetColumn.forwardArgs)[0];

        this.edgesMap[targetColumn.outModule] = {
          type: 'output',
          edges: {
            [edgeId]: [targetColumn.name],
          },
        };
      }
    });
  }

  /**
   * Get's the current list of suggestions
   *
   */
  getSuggestions = () => this.suggestions;

  /**
   * Get's shape of a node and an outgoing edge. If component outputs
   * a simple single value, outgoingEdge must be '' (empty string).
   *
   * Prefer using {@link getOutgoingShapeSimple}
   *
   * @param {string} nodeName - node identifier
   * @param {string} outgoingEdge - node output attribute
   * @returns {(number[] | undefined)} shape if known in forward order pass
   */
  getOutgoingShape = (
    nodeName: string,
    outgoingEdge: string
  ): number[] | undefined =>
    getFromMap(nodeName, outgoingEdge, this.outgoingShapes);

  getOutgoingShapeSimple = (nodeName: string): number[] | undefined => {
    const [head, ...tail] = nodeName.split('.');
    if (!head) return;
    return getFromMap(head, tail.join('.'), this.outgoingShapes);
  };

  /**
   * Get's required input matrix dimension (shape)
   *
   * Prefer using {@link getRequiredShapeSimple}
   *
   * @param {string} nodeName - node identifier
   * @param {string} outgoingEdge - node output attribute
   * @returns {(number[] | undefined)} shape if known in forward order pass
   */
  getRequiredShape = (
    nodeName: string,
    outgoingEdge: string
  ): number[] | undefined =>
    getFromMap(nodeName, outgoingEdge, this.requiredShapes);

  getRequiredShapeSimple = (nodeName: string): number[] | undefined => {
    const [head, ...tail] = nodeName.split('.');
    if (!head) return;
    return getFromMap(head, tail.join('.'), this.requiredShapes);
  };

  /**
   * Get the data type of the outgoing edge of the node with `nodeName`
   *
   * @param {string} nodeName - [TODO:description]
   * @param {string} outgoingEdge - [TODO:description]
   * @returns {(DataType | undefined)} [TODO:description]
   */
  getDataType = (
    nodeName: string,
    outgoingEdge: string
  ): DataType | undefined => getFromMap(nodeName, outgoingEdge, this.dataTypes);

  getDataTypeSimple = (nodeName: string): DataType | undefined => {
    const [head, ...tail] = nodeName.split('.');
    if (!head) return;
    return getFromMap(head, tail.join('.'), this.dataTypes);
  };

  addSuggestions = (suggestions: Suggestion[]) => {
    this.suggestions = this.suggestions.concat(suggestions);
  };

  addSuggestion = (suggestion: Suggestion) => {
    this.suggestions.push(suggestion);
  };

  setOutgoingShapeSimple(name: string, shape: number[]) {
    setMap(name, '', shape, this.outgoingShapes);
  }

  //! Not being used currently
  setOutgoingShape = (
    nodeName: string,
    outgoingEdge: string,
    shape: number[]
  ): void => {
    setMap(nodeName, outgoingEdge, shape, this.outgoingShapes);
  };

  setRequiredShapeSimple(name: string, shape: number[]) {
    setMap(name, '', shape, this.requiredShapes);
  }

  //! Not being used currently
  setRequiredShape = (
    nodeName: string,
    outgoingEdge: string,
    shape: number[]
  ): void => {
    setMap(nodeName, outgoingEdge, shape, this.requiredShapes);
  };

  setDataType = (
    nodeName: string,
    outgoingEdge: string,
    dataType: DataType
  ): void => {
    setMap(nodeName, outgoingEdge, dataType, this.dataTypes);
  };
  setDataTypeSimple = (nodeName: string, dataType: DataType): void => {
    const [head, ...tail] = nodeName.split('.');
    if (!head) return;
    setMap(nodeName, tail.join('.'), dataType, this.dataTypes);
  };

  updateInfoEdgesMap = (node: NodeType): void => {
    const dependents = getDependents(node, this.schema);

    dependents.forEach((dependent) => {
      if ('forwardArgs' in dependent && dependent.forwardArgs) {
        Object.keys(dependent.forwardArgs).forEach((edgeId) => {
          if (this.edgesMap[node.name]) {
            this.edgesMap[node.name].edges[edgeId]?.push(
              unwrapDollar(dependent.name)
            );
          } else {
            this.edgesMap[node.name] = {
              type: node.type,
              edges: {
                [edgeId]: [unwrapDollar(dependent.name)],
              },
            };
          }
        });
      }
    });
  };
}

export default TransversalInfo;

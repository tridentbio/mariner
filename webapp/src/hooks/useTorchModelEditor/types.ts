import { GetModelOptionsApiResponse } from 'app/rtk/generated/models';
import TorchModelEditorImpl from 'model-compiler/src/implementation/TorchModelEditorImpl';
import Suggestion from 'model-compiler/src/implementation/Suggestion';
import {
  LayerFeaturizerType,
  ModelSchema,
  NodeType,
} from '@model-compiler/src/interfaces/torch-model-editor';
import { Dispatch } from 'react';
import {
  Edge,
  HandleType,
  Node,
  ReactFlowInstance,
  OnNodesChange,
  OnEdgesChange,
} from 'reactflow';
import { ArrayElement } from 'utils';
import { CSSProperties } from 'styled-components';

export type MarinerNode = Node<NodeType>;

/**
 * Interface needed for TorchModelEditorV2 Context.
 * Each model editor must have a different `TorchModelEditorContextProvider`
 *
 */
export interface ITorchModelEditorContext
  extends Omit<TorchModelEditorImpl, 'addComponent' | 'applySuggestions'>,
    ReactFlowInstance<NodeType, any> {
  /**
   * Maps a model schema to react flow's graph
   *
   * @param {ModelSchema} schema - the model schema
   * @param {ReturnType<typeof getNodePositionsMap>} [positions] - optional map of node positions
   * @returns {[MarinerNode[], Edge[]]} react flow nodes and edges
   */
  schemaToEditorGraph(
    schema: ModelSchema,
    positions?: Record<string, { x: number; y: number }>
  ): [MarinerNode[], Edge[]];

  /**
   * Organizes the nodes in the editor
   */
  applyDagreLayout(
    director: 'TB' | 'LR',
    spaceValue: number,
    _edges?: Edge[]
  ): void;

  /**
   * Adds a component with a position
   * @see {@link ITorchModelEditorContext}
   */
  addComponent(
    args: {
      schema: ModelSchema;
      name: string;
      type: LayerFeaturizerType['type'];
      componentType: 'layer' | 'featurizer' | 'transformer';
    },
    position: { x: number; y: number }
  ): ModelSchema;

  /**
   * Apply suggestioned modifications to the schema
   * @see {@link ITorchModelEditorContext}
   */
  applySuggestions(
    args: Parameters<TorchModelEditorImpl['applySuggestions']>[0]
  ): ModelSchema;

  /**
   * Map a component class path to it's {@type ModelOtion}
   */
  options?: Record<string, ArrayElement<GetModelOptionsApiResponse>>;

  /**
   * Neural netcComponent options gruoped by library of origin
   */
  optionsByLib?: Record<string, GetModelOptionsApiResponse>;

  /**
   * Nodes of the graph editor
   * for details on the node attributes see {@link schemaToEditorGraph}
   */
  nodes: MarinerNode[];

  /**
   * Edges of the graph editor
   *
   * for details on the edge attributes see {@link schemaToEditorGraph}
   */
  edges: Edge<any>[];

  /**
   * Expand components constructor args form
   *
   * @param {string[]} [nodeIds] - ids of nodes to be expanded. if `undefined`
   * expand all nodes
   */
  expandNodes: (nodeIds?: string[]) => void;

  /**
   * Contract components constructor args form
   *
   * @param {string[]} [nodeIds] - ids of nodes to be expanded. if `undefined`
   * contract all nodes
   */
  contractNodes: (nodeIds?: string[]) => void;

  /**
   * Maps node id to expanded state. Node has constructorArgs form expanded if `id`
   * is present in this object
   */
  expandedNodesMap: Record<string, boolean>;

  /**
   * Current schema, updated after each model schema change
   * (through `TorchModelEditor` methods)
   */
  schema?: ModelSchema;

  /**
   * Makes a hard change to the schema (prefer using `addComponent`, `editComponent`,
   * `deleteComponents` and other methods from `TorchModelEditor` ). It's only use is to
   * set the initial state with the help of provided methods
   */
  setSchema: Dispatch<ModelSchema>;

  /**
   * Suggestions gathered from last model schema edition
   */
  suggestions: Suggestion[];

  /**
   * Maps node ids to the list of suggestions related to it
   */
  suggestionsByNode: Record<string, Suggestion[]>;

  /**
   * Maps edge ids to the list of suggestions related to it
   */
  suggestionsByEdge: Record<string, Suggestion[]>;

  /**
   * Toggles the expanded state of a node constructorArgs form
   *
   * @param {string} nodeId - id of the node to toggle
   */
  toggleExpanded: (nodeId: string) => void;

  keyAssignments?: ({ key: string; nodeId: string } & (
    | {
        targetHandleId: string;
      }
    | { sourceHandleId: string }
  ))[];

  assignPositionsOrdering: <T extends HandleType>(
    type: T,
    component: NodeType
  ) => ({ key: string; nodeId: string } & (
    | { targetHandleId: string }
    | { sourceHandleId: string }
  ))[];

  getHandleKey: (
    nodeId: string,
    type: HandleType,
    handleId: string
  ) => string | undefined;

  clearPositionOrdering: () => void;

  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;

  nodesInitialized: boolean;

  highlightNodes: (nodeIds: string[], color?: CSSProperties['color']) => void;

  readonly NODE_DEFAULT_STYLISH: {
    borderColor: string;
  };
}

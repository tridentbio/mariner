import {
  DataType,
  FeaturizersType,
  LayersType,
  ModelSchema,
} from '../../interfaces/model-editor';
import { getNodes } from '../modelSchemaQuery';
import Suggestion from '../Suggestion';

type EdgeMap<V> = {
  [key: string]: {
    [key: string]: V;
  };
};

const getFromMap = <T>(key1: string, key2: string, map: EdgeMap<T>) => {
  if (key1 in map && key2 in map[key1]) {
    return map[key1][key2];
  }
};

const setMap = <T>(key1: string, key2: string, value: T, map: EdgeMap<T>) => {
  if (!(key1 in map)) map[key1] = {};
  map[key1][key2] = value;
};

class TransversalInfo {
  suggestions: Suggestion[] = [];
  shapes: EdgeMap<number[]> = {};
  dataTypes: EdgeMap<DataType> = {};
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
   * Prefer using {@link getShapeSimple}
   *
   * @param {string} nodeName - node identifier
   * @param {string} outgoingEdge - node output attribute
   * @returns {(number[] | undefined)} shape if known in forward order pass
   */
  getShape = (nodeName: string, outgoingEdge: string): number[] | undefined =>
    getFromMap(nodeName, outgoingEdge, this.shapes);

  getShapeSimple = (nodeName: string): number[] | undefined => {
    const [head, ...tail] = nodeName.split('.');
    if (!head) return;
    return getFromMap(head, tail.join('.'), this.shapes);
  };

  /**
   * [TODO:description]
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

  setShapeSimple(name: string, shape: number[]) {
    setMap(name, '', shape, this.shapes);
  }

  setShape = (
    nodeName: string,
    outgoingEdge: string,
    shape: number[]
  ): void => {
    setMap(nodeName, outgoingEdge, shape, this.shapes);
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
}

export default TransversalInfo;

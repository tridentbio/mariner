import { expect } from '@jest/globals';
import {
  getRegressorModelSchema,
  getValidModelSchemas,
} from '../../../../tests/fixtures/model-schemas';
import { getDependencies, getDependents } from './modelSchemaQuery';

describe('modelSchemaQuery', () => {
  describe('getDependencies', () => {
    it('should return dependencies', () => {
      const schema = getRegressorModelSchema();
      if (!schema.spec.layers?.length) throw 'fail to get the model schema';
      const node = schema.spec.layers[0];
      // @ts-ignore
      const deps = getDependencies(node, schema);
      expect(deps.length).toBe(1);
      expect(deps[0].name).toBe('MolToGraphFeaturizer');
    });
  });

  describe('getDependents', () => {
    it('should return dependents', () => {
      const schema = getRegressorModelSchema();
      if (!schema.spec.layers?.length) throw 'fail to get the model schema';
      const node = schema.spec.layers[0];
      // @ts-ignore
      const deps = getDependents(node, schema);
      expect(deps.length).toBe(1);
      expect(deps[0].name).toBe('GCN1_Activation');
    });
  });
});

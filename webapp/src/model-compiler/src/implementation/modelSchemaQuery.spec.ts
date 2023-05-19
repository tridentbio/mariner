import { expect, } from '@jest/globals'
import { getRegressorModelSchema, getValidModelSchemas } from "../../../../tests/fixtures/model-schemas";
import { getDependencies } from './modelSchemaQuery';

describe('modelSchemaQuery', () => {
  describe('getDependencies', () => {
    it('should return dependencies', () => {
      const schema = getRegressorModelSchema()
      if (!schema.spec.layers?.length) throw 'fail to get the model schema'
      const node = schema.spec.layers[0]
      const deps = getDependencies(node, schema)
      expect(deps.length).toBe(1);
      expect(deps[0].name).toBe('MolToGraphFeaturizer');
    })
  })

  describe('getDependents', () => {
  })
})

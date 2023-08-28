import { expect } from '@jest/globals';
import { getRegressorModelSchema } from '../../../../tests/fixtures/model-schemas';
import { ModelSchema } from '../interfaces/torch-model-editor';
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
    it('return dependency based on target column', () => {
      const layer = {
        type: 'torch.nn.Linear',
        name: 'L1',
        constructorArgs: {
          in_features: 1,
          out_features: 1,
        },
        forwardArgs: {
          input: '$col1',
        },
      } as const;
      const schema: ModelSchema = {
        name: 'test',
        dataset: {
          name: 'test dataset',
          featureColumns: [
            {
              name: 'col1',
              dataType: {
                domainKind: 'numeric',
              },
            },
          ],
          targetColumns: [
            {
              name: 'col2',
              dataType: {
                domainKind: 'numeric',
              },
              outModule: 'L1',
            },
          ],
          featurizers: [],
          transforms: [],
        },
        spec: {
          layers: [layer],
        },
      };
      const dependents = getDependents(layer, schema);
      expect(dependents.length).toBe(1);
      expect(dependents[0].name).toBe('col2');
    });
  });
});

import { expect, test } from '@jest/globals';
import {
  BrokenSchemas,
  getRegressorModelSchema,
} from '../../../../tests/fixtures/model-schemas';
import { Linear, GcnConv, ModelSchema } from '../interfaces/model-editor';
import { makeComponentEdit } from './commands/EditComponentsCommand';
import ModelEditorImpl from './ModelEditorImpl';
import { getComponent } from './modelSchemaQuery';

const getTestModelEditor = () => {
  return new ModelEditorImpl();
};
describe('ModelEditorImpl', () => {
  describe('addComponents', () => {
    it('adds single component', () => {
      const editor = getTestModelEditor();
      const schema = getRegressorModelSchema();
      const newSchema = editor.addComponent({
        type: 'layer',
        data: {
          name: 'firstLienar',
          type: 'torch.nn.Linear',
          constructorArgs: {
            in_features: 1,
            out_features: 64,
            bias: false,
          },
          forwardArgs: {
            input: 'LinearJoined',
          },
        },
        schema,
      });
      expect(newSchema.layers).toBeDefined();
      expect(newSchema.layers?.length).toBe((schema.layers?.length || 0) + 1);
      const insertedLayer = newSchema.layers?.find(
        (l) => l.name === 'firstLienar'
      );
      expect(insertedLayer).toBeDefined();
      expect(insertedLayer?.forwardArgs).toHaveProperty('input');
      expect((insertedLayer?.forwardArgs as Linear['forwardArgs']).input).toBe(
        '$LinearJoined'
      );
    });
  });

  describe('editComponent', () => {
    it('edits a single component in the model schema', () => {
      const editor = getTestModelEditor();
      const schema = getRegressorModelSchema();
      const newSchema = editor.editComponent({
        schema,
        data: makeComponentEdit({
          component: getComponent(schema, 'GCN3') as GcnConv & {
            type: 'torch_geometric.nn.GCNConv';
          },
          constructorArgs: {
            in_channels: 1,
          },
        }),
      });

      // @ts-ignore
      const editedLayer = newSchema.layers.find(
        (layer) => layer.name === 'GCN3'
      );
      expect(editedLayer).toBeDefined();

      // @ts-ignore
      expect(editedLayer.constructorArgs.in_channels).toBe(1);
      // @ts-ignore
      expect(editedLayer.constructorArgs.out_channels).toBe(64);
    });
  });

  describe('deleteComponent', () => {
    it('deletes a single component in the model schema', () => {
      const editor = getTestModelEditor();
      const schema = getRegressorModelSchema();
      const newSchema = editor.deleteComponents({
        schema,
        nodeId: 'GCN3',
      });
      const removedLayer = newSchema.layers!.find(
        (layer) => layer.name === 'GCN3'
      );
      expect(removedLayer).toBeUndefined();
      const GCN3_GCN3Actvation_edge = Object.values(
        // @ts-ignore
        newSchema.layers!.find((layer) => layer.name === 'GCN3_Activation')
          ?.forwardArgs
      ).find((node) => node.includes('GCN3'));
      expect(GCN3_GCN3Actvation_edge).toBeUndefined();
    });
  });

  describe('getSuggestions', () => {
    it('gets suggestions for invalid schemas', () => {
      const invalidSchemas = Object.values(BrokenSchemas());
      invalidSchemas.forEach((invalidSchema) => {
        const editor = getTestModelEditor();
        const info = editor.getSuggestions({ schema: invalidSchema });
        expect(info.getSuggestions().length).toBeGreaterThan(0);
      });
    });
  });

  describe('applySuggestions', () => {
    test.each(Object.entries(BrokenSchemas()))(
      'schema BrokenSchemas.%s is fixed on applySuggestions (if fixable)',
      (_key, invalidSchema) => {
        const editor = getTestModelEditor();
        const info = editor.getSuggestions({ schema: invalidSchema });
        expect(info.schema).toMatchObject(invalidSchema);
        const gotTotalSuggestions = info.getSuggestions().length;
        const gotTotalFixableSuggestions = info
          .getSuggestions()
          .filter((s) => s.commands.length > 0).length;
        expect(gotTotalSuggestions).toBeGreaterThan(0);

        // fix suggestions
        let newSchema: ModelSchema;
        newSchema = editor.applySuggestions({
          suggestions: info.getSuggestions(),
          schema: invalidSchema,
        });

        // good model doesn't yield suggestions
        const newInfo = editor.getSuggestions({ schema: newSchema });
        expect(newInfo.getSuggestions()).toHaveLength(
          gotTotalSuggestions - gotTotalFixableSuggestions
        );
      }
    );
  });
});
import { expect, test } from '@jest/globals';
import {
  GcnConv,
  Linear,
  MolFeaturizer,
} from 'model-compiler/src/interfaces/model-editor';
import {
  BrokenSchemas,
  getValidModelSchemas,
} from '../../../../../tests/fixtures/model-schemas';

import EditComponentsCommand from '../commands/EditComponentsCommand';
import ModelValidation from './ModelValidation';

const getTestValidator = (): ModelValidation => {
  return new ModelValidation();
};

describe('ModelValidation', () => {
  describe('validate(schema)', () => {
    test.each(getValidModelSchemas())(
      'validate(%s) returns no suggestions for good schemas',
      (schema) => {
        const info = getTestValidator().validate(schema);
        expect(info.getSuggestions()).toHaveLength(0);
      }
    );

    it("returns correction to linear when in_features doesn't match incoming shape", () => {
      const info = getTestValidator().validate(
        BrokenSchemas().testLinearValidator1
      );
      expect(info.getSuggestions()).toHaveLength(2);
      const suggestion = info.getSuggestions().at(-1);
      expect(suggestion).toBeDefined();
      if (!suggestion) return;
      expect(suggestion.commands).toHaveLength(1);
      const command = suggestion?.commands[0];
      expect(command).toBeInstanceOf(EditComponentsCommand);
      if (command instanceof EditComponentsCommand) {
        expect(command.args.data.name).toBe('3');
        expect((command.args.data as Linear).constructorArgs.in_features).toBe(
          8
        );
      }
    });

    it('returns correction to mol featurizer when receiving input of data type different from smiles', () => {
      const info = getTestValidator().validate(
        BrokenSchemas().testMolFeaturizer1
      );
      expect(info.getSuggestions()).toHaveLength(1);
      const suggestion = info.getSuggestions()[0];
      expect(suggestion.commands).toHaveLength(1);
      const command = suggestion.commands[0];
      expect(command).toBeInstanceOf(EditComponentsCommand);
      if (command instanceof EditComponentsCommand) {
        expect(command.args.data.name).toBe('feat');
        expect((command.args.data as MolFeaturizer).forwardArgs.mol).toBe('');
      }
    });

    it("returns correction to gcn conv when in_channels doesn't match incoming shape", () => {
      const info = getTestValidator().validate(BrokenSchemas().testGcnConv);
      expect(info.getSuggestions()).toHaveLength(1);
      const suggestion = info.getSuggestions()[0];
      expect(suggestion.commands).toHaveLength(1);
      const command = suggestion.commands[0];
      expect(command).toBeInstanceOf(EditComponentsCommand);
      if (command instanceof EditComponentsCommand) {
        expect(command.args.data.name).toBe('1');
        expect((command.args.data as GcnConv).constructorArgs.in_channels).toBe(
          26
        );
      }
    });
  });
});

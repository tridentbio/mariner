import { expect, test } from '@jest/globals';
import { extendSpecWithTargetForwardArgs } from 'model-compiler/src/utils';
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
import { TorchModelSpec } from '@app/rtk/generated/models';
import { NodeEdgeTypes } from './TransversalInfo';

const getTestValidator = (): ModelValidation => {
  return new ModelValidation();
};

describe('ModelValidation', () => {
  describe('validate(schema)', () => {
    test.each(getValidModelSchemas())(
      'validate(%s) returns no suggestions for good schemas',
      (schema) => {
        const info = getTestValidator().validate(
          extendSpecWithTargetForwardArgs(schema)
        );
        expect(info.getSuggestions()).toHaveLength(0);
      }
    );

    it("returns correction to linear when in_features doesn't match incoming shape", () => {
      const info = getTestValidator().validate(
        extendSpecWithTargetForwardArgs(BrokenSchemas().testLinearValidator1)
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

    it("return correction to linear when out_features doesn't match next layer dimension restriction", () => {
      // Test numeric output
      const spec: TorchModelSpec = {
        name: 'test',
        dataset: {
          name: 'test-ds',
          featureColumns: [
            {
              name: 'col1',
              dataType: {
                unit: 'mole',
                domainKind: 'numeric',
              },
            },
          ],
          targetColumns: [
            {
              name: 'col2',
              dataType: {
                unit: 'mole',
                domainKind: 'numeric',
              },
              outModule: 'L1',
              lossFn: 'torch.nn.MSELoss',
            },
          ],
          featurizers: [],
        },
        spec: {
          layers: [
            {
              type: 'torch.nn.Linear',
              name: 'L1',
              forwardArgs: {
                input: '$col1',
              },
              constructorArgs: {
                in_features: 1,
                out_features: 24, // should be 1 since it's final layer
              },
            },
          ],
        },
      };
      const info = getTestValidator().validate(
        extendSpecWithTargetForwardArgs(spec)
      );
      expect(info.getSuggestions()).toHaveLength(1);
      const suggestion = info.getSuggestions()[0];
      expect(suggestion.commands).toHaveLength(1);
      const command = suggestion.commands[0];
      expect(command).toBeInstanceOf(EditComponentsCommand);
      if (command instanceof EditComponentsCommand) {
        expect(command.args.data.name).toBe('L1');
        expect((command.args.data as Linear).constructorArgs.out_features).toBe(
          1
        );
      }

      // Test categorical output with 3 classes
      const spec2: TorchModelSpec = {
        name: 'test',
        dataset: {
          name: 'test-ds',
          featureColumns: [
            {
              name: 'col1',
              dataType: {
                unit: 'mole',
                domainKind: 'numeric',
              },
            },
          ],
          targetColumns: [
            {
              name: 'col2',
              dataType: {
                domainKind: 'categorical',
                classes: {
                  a: 0,
                  b: 1,
                  c: 2,
                },
              },
              outModule: 'L1',
              lossFn: 'torch.nn.CrossEntropyLoss',
            },
          ],
          featurizers: [],
        },
        spec: {
          layers: [
            {
              type: 'torch.nn.Linear',
              name: 'L1',
              forwardArgs: {
                input: '$col1',
              },
              constructorArgs: {
                in_features: 1,
                out_features: 24, // should be 3 since it's final layer
              },
            },
          ],
        },
      };
      const info2 = getTestValidator().validate(
        extendSpecWithTargetForwardArgs(spec2)
      );
      expect(info2.getSuggestions()).toHaveLength(1);
      const suggestion2 = info2.getSuggestions()[0];
      expect(suggestion2.commands).toHaveLength(1);
      const command2 = suggestion2.commands[0];
      expect(command2).toBeInstanceOf(EditComponentsCommand);
      if (command2 instanceof EditComponentsCommand) {
        expect(command2.args.data.name).toBe('L1');
        expect(
          (command2.args.data as Linear).constructorArgs.out_features
        ).toBe(3);
      }
    });

    it('returns correction to mol featurizer when receiving input of data type different from smiles', () => {
      const info = getTestValidator().validate(
        extendSpecWithTargetForwardArgs(BrokenSchemas().testMolFeaturizer1)
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
      const info = getTestValidator().validate(
        extendSpecWithTargetForwardArgs(BrokenSchemas().testGcnConv)
      );
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

  it('should mount the TransversalInfo edgesMap attribute properly with "one to many" node association', () => {
    const info = getTestValidator().validate(
      extendSpecWithTargetForwardArgs(
        BrokenSchemas().testOneToManyEdgeAssociation
      )
    );

    const rootNodeEdgeMap = info.edgesMap[
      'rootNode'
    ] as NodeEdgeTypes<'torch.nn.Linear'>;

    expect(rootNodeEdgeMap?.type).toBe('torch.nn.Linear');
    expect(rootNodeEdgeMap?.edges).toBeDefined();
    expect(rootNodeEdgeMap?.edges?.input).toHaveLength(3);
    expect(rootNodeEdgeMap?.edges?.input).toContain('childNode1');
    expect(rootNodeEdgeMap?.edges?.input).toContain('childNode2');
    expect(rootNodeEdgeMap?.edges?.input).toContain('childNode3');
  });
});

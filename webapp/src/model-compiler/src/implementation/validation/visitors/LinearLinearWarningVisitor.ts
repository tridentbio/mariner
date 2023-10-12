import { LayersType } from '@model-compiler/src/interfaces/torch-model-editor';
import ComponentVisitor from './ComponentVisitor';
import { getDependents } from '../../modelSchemaQuery';
import Suggestion from '../../Suggestion';
import AddComponentCommand from '../../commands/AddComponentCommand';
import EditComponentsCommand from '../../commands/EditComponentsCommand';

export default class LinearLinearWarningVisitor extends ComponentVisitor {
  visitLinear: ComponentVisitor['visitLinear'] = ({ info, component }) => {
    const edgeMap = info.edgesMap[component.name];

    if (!edgeMap) return;

    const dependents = getDependents(component, info.schema);

    let dependentLinearLayers = dependents.filter(
      (dependent) => dependent.type == component.type
    ) as (LayersType & { type: 'torch.nn.Linear' })[];

    dependentLinearLayers.forEach((dependentLinearLayer) => {
      const nonLinearLayer: LayersType = {
        name: `${dependentLinearLayer.name}-${component.name}-ReLu`,
        type: 'torch.nn.ReLU',
        constructorArgs: { inplace: false },
        forwardArgs: {
          input: component.name,
        },
      };

      info.addSuggestion(
        new Suggestion(
          'WARNING',
          'Sequential Linear layers are not recommended without a nonlinear layer',
          [
            new AddComponentCommand({
              type: 'layer',
              data: nonLinearLayer,
              position: {
                type: 'relative',
                references: [dependentLinearLayer.name, component.name],
              },
            }),
            new EditComponentsCommand({
              data: (schema) => {
                const foundSpec = schema.spec.layers?.find(
                  (node) => node.name == dependentLinearLayer.name
                );

                if (!foundSpec) {
                  // eslint-disable-next-line no-console
                  console.error('Could not find dependent linear layer');
                  return null as any;
                }

                foundSpec.forwardArgs = { input: nonLinearLayer.name };

                return foundSpec;
              },
            }),
          ],
          {
            nodeId: component.name,
            edgeId: `${dependentLinearLayer.name}-${component.name}`,
          }
        )
      );
    });
  };
}

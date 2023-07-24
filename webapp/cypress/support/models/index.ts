import { buildModel, buildYamlModel } from './build-model';
import { setupSomeModel, modelFormData } from './create';

Cypress.Commands.add('buildYamlModel', buildYamlModel);
Cypress.Commands.add('setupSomeModel', setupSomeModel);

declare global {
  namespace Cypress {
    interface Chainable {
      buildYamlModel(
        yaml: string,
        datasetName?: string,
        buildParams?: Parameters<typeof buildModel>[1],
        modelName?: string
      ): Chainable<void>;
      setupSomeModel(): Chainable<ReturnType<typeof modelFormData>>;
    }
  }
}

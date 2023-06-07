import { buildYamlModel, setupSomeModel } from './build-model';

Cypress.Commands.add('buildYamlModel', buildYamlModel);
Cypress.Commands.add('setupSomeModel', setupSomeModel);

declare global {
  namespace Cypress {
    interface Chainable {
      buildYamlModel(
        yaml: string,
        datasetName?: string,
        success?: boolean,
        deleteModel?: boolean,
        modelName?: string
      ): Chainable<void>;
      setupSomeModel(): Chainable<string>;
    }
  }
}

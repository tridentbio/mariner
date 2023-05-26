import { buildYamlModel } from './build-model';

Cypress.Commands.add('buildYamlModel', buildYamlModel);

declare global {
  namespace Cypress {
    interface Chainable {
      buildYamlModel(
        yaml: string,
        datasetName?: string,
        success?: boolean,
        deleteModel?: boolean,
        modelName?: string
      ): string;
    }
  }
}

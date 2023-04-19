import { buildYamlModel } from './smiles-num';
import { DatasetFormData } from '../dataset/create';

Cypress.Commands.add('buildYamlModel', buildYamlModel);

declare global {
  namespace Cypress {
    interface Chainable {
      buildYamlModel(
        yaml: string,
        datasetName?: string,
        success?: boolean
      ): string;
    }
  }
}

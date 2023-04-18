import { buildNumSmilesModel, buildYamlModel } from './smiles-num';
import { buildCategoricalSmilesModel } from './smiles-categorical';
import { DatasetFormData } from '../dataset/create';

Cypress.Commands.add('buildNumSmilesModel', buildNumSmilesModel);
Cypress.Commands.add(
  'buildCategoricalSmilesModel',
  buildCategoricalSmilesModel
);
Cypress.Commands.add('buildYamlModel', buildYamlModel);

declare global {
  namespace Cypress {
    interface Chainable {
      buildNumSmilesModel(dataset?: string): Chainable;
      buildCategoricalSmilesModel(
        featureCols: string[],
        targetCol: string,
        dataset?: string
      ): Chainable;
      buildYamlModel(yaml: string, datasetName?: string): Chainable;
    }
  }
}

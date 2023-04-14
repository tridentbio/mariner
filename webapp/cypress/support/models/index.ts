import { buildNumSmilesModel } from './smiles-num';
import { buildCategoricalSmilesModel } from './smiles-categorical';
import { DatasetFormData } from '../dataset/create';

Cypress.Commands.add('buildNumSmilesModel', buildNumSmilesModel);
Cypress.Commands.add(
  'buildCategoricalSmilesModel',
  buildCategoricalSmilesModel
);

declare global {
  namespace Cypress {
    interface Chainable {
      buildNumSmilesModel(dataset?: string): Chainable;
      buildCategoricalSmilesModel(
        featureCols: string[],
        targetCol: string,
        dataset?: string
      ): Chainable;
    }
  }
}

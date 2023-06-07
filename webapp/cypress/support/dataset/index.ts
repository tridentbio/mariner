import {
  DatasetFormData,
  createDatasetDirectly,
  useIrisDataset,
} from './create';

Cypress.Commands.add('createDatasetDirectly', createDatasetDirectly);
Cypress.Commands.add('useIrisDataset', useIrisDataset);

declare global {
  namespace Cypress {
    interface Chainable {
      createDatasetDirectly(
        dataset: DatasetFormData,
        numRows?: number // default=10
      ): void;
      useIrisDataset(): Cypress.Chainable<{
        setup: () => void;
        fixture: DatasetFormData;
      }>;
    }
  }
}

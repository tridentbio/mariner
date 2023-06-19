import { CyHttpMessages } from 'cypress/types/net-stubbing';
import {
  createDeployment,
  updateDeployment,
  runAction,
  getDeploymentForm,
} from './create-update';
import { deleteDeployment } from './delete';
import {
  goToDeploymentWithinModel,
  goToPublicDeployment,
  goToSpecificDeployment,
  openDeploymentInCurrentTable,
} from './find-deployment';
import {
  handleStatus,
  makePrediction,
  startDeployment,
  stopDeployment,
} from './workflow';

Cypress.Commands.add('createDeployment', createDeployment);
Cypress.Commands.add('updateDeployment', updateDeployment);
Cypress.Commands.add('runAction', runAction);

Cypress.Commands.add('deleteDeployment', deleteDeployment);

Cypress.Commands.add('goToSpecificDeployment', goToSpecificDeployment);
Cypress.Commands.add('goToDeploymentWithinModel', goToDeploymentWithinModel);
Cypress.Commands.add('goToPublicDeployment', goToPublicDeployment);
Cypress.Commands.add(
  'openDeploymentInCurrentTable',
  openDeploymentInCurrentTable
);

Cypress.Commands.add('startDeployment', startDeployment);
Cypress.Commands.add('stopDeployment', stopDeployment);
Cypress.Commands.add('handleStatus', handleStatus);
Cypress.Commands.add('makePrediction', makePrediction);

declare global {
  namespace Cypress {
    interface Chainable {
      createDeployment(
        modelName: string,
        modelVersionName: string
      ): Cypress.Chainable<string>;
      updateDeployment(
        modelName: string,
        modelVersionName: string,
        deploymentFormData: Partial<ReturnType<typeof getDeploymentForm>>
      ): Cypress.Chainable<CyHttpMessages.IncomingResponse | undefined>;
      runAction(deploymentName: string, action: number): void;

      deleteDeployment(modelName: string, deploymentName: string): void;

      goToSpecificDeployment(
        name: string,
        tab?: 'All' | 'Public' | 'Shared' | 'My'
      ): void;
      goToDeploymentWithinModel(modelName: string): void;
      goToPublicDeployment(shareUrl: string): void;
      openDeploymentInCurrentTable(deploymentName: string): void;

      startDeployment(deploymentName: string): void;
      stopDeployment(deploymentName: string): void;
      handleStatus(deploymentName: string, status: 'active' | 'idle'): void;
      makePrediction(expectSuccess: boolean): void;
    }
  }
}

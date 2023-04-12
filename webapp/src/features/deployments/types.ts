import { ModelVersion } from 'app/types/domain/models';
import { PaginationQueryParams } from 'app/types/paginationQuery';
import { User } from 'features/users/usersAPI';

export interface Deployment {
  id: number;
  name: string;
  readme: string;
  shareUrl: string;
  status: EDeploymnetStatuses;
  modelVersionId: number;
  modelVersion?: ModelVersion;
  shareStrategy: EShareStrategies;
  usersIdAllowed: User['id'][];
  usersAllowed?: User[];
  organizationsAllowed: string[]; // check email path
  showTrainingData: boolean;
  predictionRateLimitValue: number;
  predictionRateLimitUnit: ERateLimitUnits;
  createdByUserId: User['id'];
  createdAt: Date;
  isDeleted: boolean;
}

export enum EDeploymnetStatuses {
  ACTIVE = 'active',
  STOPPED = 'stopped',
  STARTING = 'starting',
  IDLE = 'idle',
}
export enum EShareStrategies {
  PRIVATE = 'private',
  PUBLIC = 'public',
}
export enum ERateLimitUnits {
  MINUTE = 'minute',
  HOUR = 'hour',
  DAY = 'day',
  MONTH = 'month',
}

export interface DeploymentsQuery extends PaginationQueryParams {
  name?: string;
  status?: EDeploymnetStatuses;
  shareStrategy?: EShareStrategies;
  createdAfter?: Date;
  modelVersionId?: number;
}

export interface DeploymentCreateRequest
  extends Omit<
    Deployment,
    | 'isDeleted'
    | 'createdAt'
    | 'status'
    | 'usersAllowed'
    | 'shareUrl'
    | 'createdByUserId'
  > {}
export interface DeploymentUpdateRequest
  extends Partial<DeploymentCreateRequest> {
  deploymentId: number;
}

export interface DeploymentsState {
  deployments: Deployment[];
  current?: Deployment;
  totalDeployments: number;
}

export interface DeploymentFormFields
  extends Omit<
    Deployment,
    | 'status'
    | 'modelVersion'
    | 'usersAllowed'
    | 'createdByUserId'
    | 'createdAt'
    | 'isDeleted'
    | 'shareUrl'
  > {}
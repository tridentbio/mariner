import {
  SklearnTrainingRequest,
  TorchTrainingRequest,
} from '@app/rtk/generated/experiments';
import { User } from 'app/rtk/auth';
import { Model, ModelVersion } from 'app/rtk/generated/models';

export interface FetchExperimentsQuery {
  modelId?: number;
  page?: number;
  perPage?: number;
  orderBy?: string; //example: +createdAt,-name
  stage?: string[];
  modelVersionIds?: number[];
}

export type MetricMode = 'min' | 'max';

export interface NewTraining {
  modelVersionId: number;
  batchSize: number;
  name: string;
  learningRate: number;
  epochs: number;
  checkpointConfig: {
    metricKey: string;
    mode: MetricMode;
  };
  earlyStoppingConfig?: {
    metricKey: string;
    minDelta?: number;
    stopPatience?: number;
    mode?: MetricMode;
    assertFiniteMetric?: boolean;
  };
}

export type Metrics = {
  [key: string]: number;
};
export type HistoryMetrics = {
  [key: string]: number[];
};

export interface ExperimentHistory {
  experimentId: number;
  experimentName: string;
  runningHistory: HistoryMetrics;
}

export type TrainingStage = 'RUNNING' | 'SUCCESS' | 'NOT RUNNING' | 'ERROR';
export interface Experiment {
  modelVersion: ModelVersion;
  model?: Model;
  modelId?: number;
  id: number;
  modelVersionId: number;
  mlflowId: string;
  experimentName?: string;
  /**
   * A float between 0 and 1 that represents the
   * time to finish training the model.
   * The time is calculated as currentEpoch/epochs
   * */
  progress?: number;
  createdAt: Date;
  updatedAt: Date;
  stage: TrainingStage;
  createdBy: User;
  hyperparams?: Metrics;
  trainMetrics?: Metrics;
  epochs?: number;
  testMetrics?: Metrics;
  valMetrics?: Metrics;
  history?: HistoryMetrics;
  stackTrace?: string;
}

export type BaseTrainingRequest = TorchTrainingRequest | SklearnTrainingRequest;

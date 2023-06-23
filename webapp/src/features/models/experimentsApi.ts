import api, { Paginated } from 'app/api';
import {
  Experiment,
  ExperimentHistory,
  NewTraining,
  FetchExperimentsQuery,
} from 'app/types/domain/experiments';

export const startTraning = async (
  experiment: NewTraining
): Promise<Experiment> => {
  return api
    .post('/v1/experiments/', {
      modelVersionId: experiment.modelVersionId,
      name: experiment.name,
      learningRate: experiment.learningRate,
      epochs: experiment.epochs,
    })
    .then((res) => res.data);
};

export const fetchExperiments = async (
  query: FetchExperimentsQuery
): Promise<Paginated<Experiment>> => {
  return api
    .get('/v1/experiments/', {
      params: query,
    })
    .then((res) => res.data);
};

export const fetchRunningExperimentsHistory = async (): Promise<
  ExperimentHistory[]
> => {
  return api.get('/v1/experiments/running-history').then((res) => res.data);
};

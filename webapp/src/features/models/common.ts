import { Experiment } from 'app/types/domain/experiments';

type SampleExperimentReturn = {
  successful: boolean;
  running: boolean;
  failed: boolean;
  notstarted: boolean;
};

export const sampleExperiment = (
  trainings: Experiment[]
): SampleExperimentReturn => {
  const initialState = {
    successful: false,
    running: false,
    failed: false,
    notstarted: false,
  };
  return trainings.reduce((acc, current) => {
    acc = {
      successful: current.stage === 'SUCCESS' || acc.successful,
      running: current.stage === 'RUNNING' || acc.running,
      notstarted: current.stage === 'NOT RUNNING' || acc.notstarted,
      failed: current.stage === 'ERROR' || acc.failed,
    };

    return acc;
  }, initialState as SampleExperimentReturn);
};

export const extractVal = (val: any): number => {
  const guards = {
    isNumber: (val: any): val is number => typeof val === 'number',
    isListOfNumber: (val: any): val is number[] =>
      'length' in val && typeof val[0] === 'number',
    isListOfListOfNumber: (val: any): val is number[][] =>
      val?.length && val[0]?.length && typeof val[0][0] === 'number',
  };
  if (guards.isNumber(val)) return val;
  else if (guards.isListOfNumber(val)) return val[0];
  else if (guards.isListOfListOfNumber(val)) return val[0][0];
  return 0;
};

import { BaseTrainingRequest } from '@app/rtk/generated/experiments';
import { MetricMode } from 'app/types/domain/experiments';
import { DeepPartial } from 'react-hook-form';

const defaultExperimentFormValues: DeepPartial<BaseTrainingRequest> = {
  config: {
    batchSize: 32,
    optimizer: {
      classPath: 'torch.optim.Adam',
      params: {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 0,
      },
    },
    epochs: 100,
    checkpointConfig: { mode: 'min' as MetricMode },
    earlyStoppingConfig: {
      mode: 'min' as MetricMode,
      minDelta: 0,
      patience: 10,
    },
  },
};

export default defaultExperimentFormValues;

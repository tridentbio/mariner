import { Experiment } from '@app/types/domain/experiments';
import { DeepPartial } from '@reduxjs/toolkit';

export const rows: DeepPartial<Experiment>[] = [
  {
    id: 1,
    experimentName: 'Test 1',
    stage: 'SUCCESS',
    trainMetrics: {
      'val/loss/price': 0.1,
      'train/loss/price': 0.2,
    },
    hyperparams: {
      learning_rate: 0.001,
    },
    epochs: 9,
    modelVersion: {
      config: {
        dataset: {
          targetColumns: [{
            name: 'price',
          }]
        }
      }
    },
    //@ts-ignore
    createdAt: '2023-05-12'
  },
  {
    id: 2,
    experimentName: 'Test 2',
    stage: 'ERROR',
    modelVersion: {
      config: {
        dataset: {
          targetColumns: [{
            name: 'large_petal_length',
          }]
        }
      }
    },
    trainMetrics: {
      "train/accuracy/large_petal_length": 0,
      "train/precision/large_petal_length": 0,
      "train/recall/large_petal_length": 0,
      "train/f1/large_petal_length": 0,
      "train/loss/large_petal_length": 0.6939458250999451,
      "val/loss/large_petal_length": 0.24353459453582764,
    },
    hyperparams: {},
    epochs: 15,
    //@ts-ignore
    createdAt: '2023-06-23'
  },
]
import TexMath from 'components/atoms/TexMath';
import { ReactNode } from 'react';

export const trainingMetricsOptions: { key: string; label: ReactNode }[] = [
  {
    key: 'train_loss',
    label: 'Train Loss',
  },
  {
    key: 'train_mse',
    label: 'Train MSE',
  },
  {
    key: 'train_mae',
    label: 'Train MAE',
  },
  {
    key: 'train_ev',
    label: 'Train EV',
  },
  {
    key: 'train_mape',
    label: 'Train MAPE',
  },
  {
    key: 'train_R2',
    label: (
      <span>
        Train <TexMath tex="R^2" />
      </span>
    ),
  },
  {
    key: 'train_pearson',
    label: 'Train Pearson',
  },
  // removed task: UI fixes - 1/25/2023
  // {
  //   key: 'epoch',
  //   label: 'Epoch',
  // },
];

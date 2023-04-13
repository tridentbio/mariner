import { Chip } from '@mui/material';
import { Experiment } from '@app/types/domain/experiments';
import { sampleExperiment } from '../common';

export interface TrainingChipProps {
  trainings: Experiment[];
}

const TrainingStatusChip = ({ trainings }: TrainingChipProps) => {
  if (!trainings) return <Chip label="Untrained" color="warning" />;
  const { successful, running, failed, notstarted } =
    sampleExperiment(trainings);
  if (successful) return <Chip label="Trained" color="success" />;
  else if (running) return <Chip label="Running" />;
  else if (failed) return <Chip label="Failed" color="error" />;
  else if (notstarted) return <Chip label="Not started" />;
  return null;
};

export default TrainingStatusChip;

import { Button, Container } from '@mui/material';
import { ModelConfig } from 'app/types/domain/models';
import React from 'react';
import { Edge, Node, useEdges, useNodes } from 'react-flow-renderer';

type ModelCreateActionButtonsProps = {
  steps: ('Dataset' | 'ModelArch')[];
  activeStep: 'Dataset' | 'ModelArch';
  setActiveStep: React.Dispatch<React.SetStateAction<'Dataset' | 'ModelArch'>>;
  validateDatasetStep: () => ModelConfig['dataset'] | undefined;
  processing: boolean;
  handleCreateModel: ({
    edges,
    nodes,
  }: {
    nodes: Node<any>[];
    edges: Edge<any>[];
  }) => void;
};

const ModelCreateActionButtons: React.FC<ModelCreateActionButtonsProps> = ({
  activeStep,
  steps,
  setActiveStep,
  validateDatasetStep,
  processing,
  handleCreateModel,
}) => {
  const nodes = useNodes();
  const edges = useEdges();

  const firstStep = 'Dataset';
  const lastStep = 'ModelArch';
  const handleNext = () => {
    const dataset = validateDatasetStep();
    if (activeStep === 'Dataset' && !dataset) {
      return;
    }
    const currentIndex = steps.indexOf(activeStep);
    setActiveStep(steps[currentIndex + 1]);
  };
  const handlePrevious = () => {
    const currentIndex = steps.indexOf(activeStep);
    setActiveStep(steps[currentIndex - 1]);
  };

  return (
    <Container>
      {lastStep === activeStep && (
        <Button onClick={handlePrevious} variant="contained">
          PREVIOUS
        </Button>
      )}
      {firstStep === activeStep && (
        <Button sx={{ mt: 2 }} onClick={handleNext} variant="contained">
          NEXT
        </Button>
      )}
      {lastStep === activeStep && (
        <Button
          sx={{ ml: 3 }}
          disabled={processing}
          variant="contained"
          onClick={() => handleCreateModel({ nodes, edges })}
        >
          CREATE MODEL
        </Button>
      )}
    </Container>
  );
};

export { ModelCreateActionButtons };

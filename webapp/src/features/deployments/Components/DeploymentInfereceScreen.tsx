import { DeploymentWithTrainingData } from '@app/rtk/generated/deployments';
import { Section } from '@components/molecules/Section';
import { Box, SxProps } from '@mui/material';
import TorchModelEditorMarkdown from '@utils/codeSplittingAux/TorchModelEditorMarkdown';
import { DeploymentPrediction } from '@components/templates/DeploymentPrediction';
import DataSummary, {
  DataSummaryProps,
} from '@features/models/components/ModelVersionInferenceView/DataSummary';

const readmeSx: SxProps = {
  mb: '1rem',
  mt: '1rem',
  ml: '5px',
  border: '1px solid rgba(0, 0, 0, 0.12)',
  padding: '1rem',
  borderRadius: '4px',
};

export const DeploymentInferenceScreen = ({
  deployment,
  publicDeployment = false,
}: {
  deployment: DeploymentWithTrainingData;
  publicDeployment?: boolean;
}) => (
  <>
    <Section title="README">
      <Box sx={readmeSx}>
        <TorchModelEditorMarkdown
          source={deployment.readme}
          warpperElement={{
            'data-color-mode': 'light',
          }}
        />
      </Box>
    </Section>
    <DeploymentPrediction
      deployment={deployment}
      publicDeployment={publicDeployment}
    />
    {deployment?.datasetSummary && (
      <Section title="Training Data">
        <DataSummary
          columnsData={
            deployment.datasetSummary as DataSummaryProps['columnsData']
          }
        />
      </Section>
    )}
  </>
);

import { Box } from '@mui/system';
import { useState } from 'react';
import { Model, ModelVersion } from 'app/types/domain/models';
import ModelVersionInferenceView from './ModelVersionInferenceView';
import ModelVersionSelect from './ModelVersionSelect';

interface ModelInferenceViewProps {
  model: Model;
}
const ModelInferenceView = ({ model }: ModelInferenceViewProps) => {
  const [version, setVersion] = useState<ModelVersion | undefined>();

  return (
    <Box>
      <ModelVersionSelect value={version} onChange={setVersion} model={model} />
      {version && (
        <ModelVersionInferenceView model={model} modelVersionId={version.id} />
      )}
    </Box>
  );
};

export default ModelInferenceView;

import Content from 'components/templates/AppLayout/Content';
import { useMatch } from 'react-router-dom';
import DatasetEditView from '../components/DatasetEditView';

const DatasetEdit = () => {
  const path = '/datasets/:id/edit' as const;
  const match = useMatch(path);
  return (
    <Content>
      <DatasetEditView id={match?.params.id as string} />
    </Content>
  );
};

export default DatasetEdit;

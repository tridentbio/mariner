import Content from 'components/templates/AppLayout/Content';
import { useMatch } from 'react-router-dom';
import DatasetDetailsView from '../components/DatasetDetailsView';

const DatasetDetails = () => {
  const path = '/datasets/:id' as const;
  const match = useMatch(path);
  return (
    <Content>
      <DatasetDetailsView id={match?.params.id as string} />
    </Content>
  );
};

export default DatasetDetails;

import { TopBar } from '@components/organisms/TopBar';
import { Outlet } from 'react-router-dom';
import styled from 'styled-components';

const AppContainer = styled.div`
  min-height: 100vh;
`;

export function AppLayout() {
  return (
    <AppContainer>
      <TopBar />
      <Outlet />
    </AppContainer>
  );
}

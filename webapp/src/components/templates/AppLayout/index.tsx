import Footer from '@components/molecules/Footer';
import { TopBar } from '@components/organisms/TopBar';
import { Outlet } from 'react-router-dom';
import styled from 'styled-components';

const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 100vh;
`;

export function AppLayout() {
  return (
    <AppContainer>
      <TopBar />
      <Outlet />
      <Footer />
    </AppContainer>
  );
}

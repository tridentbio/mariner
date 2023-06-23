import { TopBar } from 'components/organisms/TopBar';
import { Outlet } from 'react-router-dom';

export function AppLayout() {
  return (
    <>
      <TopBar />
      <Outlet />
    </>
  );
}

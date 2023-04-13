import { Breadcrumbs as MuiBreadcrumbs } from '@mui/material';
import AppLink from 'components/atoms/AppLink';
import { Text } from 'components/molecules/Text';
import useAppNavigation from '@hooks/useAppNavigation';

const Breadcrumbs = () => {
  const { breadcrumbs } = useAppNavigation();

  return (
    <MuiBreadcrumbs>
      {breadcrumbs.map(({ label, url }, idx) =>
        idx !== breadcrumbs.length - 1 ? (
          <AppLink to={url} key={url}>
            {label}
          </AppLink>
        ) : (
          <Text key={url}>{label}</Text>
        )
      )}
    </MuiBreadcrumbs>
  );
};
export default Breadcrumbs;

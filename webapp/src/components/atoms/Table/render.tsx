import { formatDistanceToNow } from 'date-fns';
import AppLink from 'components/atoms/AppLink';

export const dateRender = <T extends {}>(getDate: (thing: T) => Date) => {
  return (row: T) =>
    formatDistanceToNow(new Date(getDate(row)), {
      addSuffix: true,
    });
};

export const linkRender = <T extends {}>(
  getLink: (thing: T) => string,
  getLabel: (thing: T) => string
) => {
  const render = (row: T) => (
    <AppLink to={getLink(row)}>{getLabel(row)}</AppLink>
  );

  render.displayName = 'AppLink';

  return render;
};

import DoneIcon from '@mui/icons-material/Done';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import { CircularProgress } from '@mui/material';
import styled from 'styled-components';

const Container = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

type IIconsMap = {
  ready: JSX.Element;
  failed: JSX.Element;
  processing: JSX.Element;
};

const iconsMap = {
  ready: <DoneIcon sx={{ color: 'success.main' }} />,
  failed: <ErrorOutlineIcon sx={{ color: 'error.main' }} />,
  processing: <CircularProgress size={25} />,
};

export const StatusButton = ({ status }: { status: keyof IIconsMap }) => {
  if (!Object.keys(iconsMap).includes(status)) return <></>;

  const capitalized = status.replace(
    /(\w)(.+)/g,
    (m: string, g: string) => g.toUpperCase() + m.slice(1)
  );

  const icon = iconsMap[status];

  return (
    <Container>
      {icon}
      {capitalized}
    </Container>
  );
};

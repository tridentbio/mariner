import styled from 'styled-components';
import GitHubIcon from '@mui/icons-material/GitHub';
import { Button, ButtonProps, Divider, Typography } from '@mui/material';

const GITHUB_COLOR = '#444444';
const GithubButtonContainer = styled(Button)`
  background-color: ${GITHUB_COLOR};
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: flex-start;
  &:hover {
    background-color: #0a0a0a;
  }
  color: white;

  text-transform: none;
  p {
    font-weight: 200;
  }
`;
const GithubButton = (props: ButtonProps) => {
  return (
    <GithubButtonContainer {...props}>
      <GitHubIcon sx={{ mr: 1 }} />
      <Divider
        flexItem
        light
        orientation={'vertical'}
        sx={{ mr: 1, backgroundColor: 'white' }}
      />
      <Typography>Sign in with GitHub</Typography>
    </GithubButtonContainer>
  );
};

export default GithubButton;

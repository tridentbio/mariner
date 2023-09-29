import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import {
  Stack,
  TextField,
  FormControl,
  Button,
  Alert,
  Typography,
} from '@mui/material';
import { LargerBoldText } from 'components/molecules/Text';
import { isApiError, messageApiError } from 'app/rtk/api';
import { authApi } from 'app/rtk/auth';
import { Box } from '@mui/system';
import { ELocalStorage } from 'app/local-storage';
import GithubButton from 'components/molecules/GithubButton';
import Logo from 'components/atoms/Logo';
import { useNotifications } from 'app/notifications';
import { getProviders, Provider } from './usersAPI';

const makeOAuthUrl = (provider: string) =>
  `${import.meta.env.VITE_API_BASE_URL}/api/v1/oauth?provider=${provider}`;

const stack = (props: { children: React.ReactNode }) => {
  return (
    <Stack
      sx={{
        '& > *': { marginTop: 3 },
        boxShadow: 'rgba(0,0,0,0.24) 0px 3px 8px',
        backgroundColor: 'white',
        borderRadius: 10,
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 5px 20px 5px',
        minWidth: 399,
        maxWidth: 599,
        height: 400,
        flexDirection: 'column',
        display: 'flex',
      }}
    >
      {props.children}
    </Stack>
  );
};

const AuthenticationPage = function () {
  const [login, { data, error, isLoading, isSuccess }] =
    authApi.useLoginMutation();
  const [providers, setProviders] = useState<Provider[]>();
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();
  const tk = params.get('tk');
  const githubError = params.get('error');
  const { notifyError } = useNotifications();

  const afterLogin =
    (location?.state as { from?: Location })?.from?.pathname || '/';
  useEffect(() => {
    getProviders().then(setProviders);
  }, []);
  useEffect(() => {
    if (githubError) {
      notifyError(githubError);
    }
  }, []);

  const handleLoginSubmit: React.FormEventHandler = async (event) => {
    event.preventDefault();
    await login({
      username: formValues.email,
      password: formValues.password,
    });
  };

  useEffect(() => {
    if (tk) {
      localStorage.setItem(
        ELocalStorage.TOKEN,
        JSON.stringify({ access_token: tk, token_type: 'bearer' })
      );
      navigate(afterLogin, { replace: true });
    }
  }, [tk]);

  useEffect(() => {
    if (isSuccess) {
      navigate(afterLogin, { replace: true });
    }
  }, [isSuccess, data]);

  const [formValues, setFormValues] = useState({
    email: '',
    password: '',
  });

  const handleFormChange =
    (
      field: keyof typeof formValues
    ): React.ChangeEventHandler<HTMLInputElement> =>
    (event) => {
      event.preventDefault();
      setFormValues({
        ...formValues,
        [field]: event.target.value,
      });
    };

  return (
    <Box
      sx={{
        flexDirection: 'column',
        height: '100vh',
        width: '100vw',
        backgroundColor: 'primary.light',
        justifyContent: 'center',
        alignItems: 'center',
        display: 'flex',
      }}
    >
      <Logo sx={{ mb: 3 }} />
      <form onSubmit={handleLoginSubmit}>
        <FormControl component={stack}>
          <LargerBoldText>Sign in</LargerBoldText>
          {error && (
            <Alert severity="error">
              {isApiError(error) ? messageApiError(error) : 'Internal Error'}
            </Alert>
          )}
          <TextField
            id="username-input"
            label="Email"
            onChange={handleFormChange('email')}
          />
          <TextField
            id="password-input"
            type="password"
            label="Password"
            onChange={handleFormChange('password')}
          />
          <Button
            disabled={isLoading || !formValues.email || !formValues.password}
            variant="contained"
            type="submit"
          >
            Sign In
          </Button>
          {providers &&
            providers.map(({ name, id }) => (
              <a
                key={id}
                style={{ textDecoration: 'none' }}
                href={makeOAuthUrl(id)}
              >
                {id === 'github' && <GithubButton />}
                {id !== 'github' && (
                  <Button variant="contained">
                    <Typography>Sign in with {name}</Typography>
                  </Button>
                )}
              </a>
            ))}
        </FormControl>
      </form>
    </Box>
  );
};

export default AuthenticationPage;

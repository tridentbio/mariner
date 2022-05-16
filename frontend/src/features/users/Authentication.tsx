import React, { useState} from 'react'
import {Link, useLocation, useNavigate} from 'react-router-dom'
import { Stack, TextField, FormControl, Button, Alert } from '@mui/material'
import {Text, LargerBoldText} from '../../components/Text'
import { useAppDispatch } from '../../app/hooks'
import {login} from './usersSlice'

const stack =  (props: { children: React.ReactNode }) => {
  return <Stack display="flex" flexDirection="column" height={500} justifyContent="space-around" maxWidth={599} marginLeft="auto" marginRight="auto"> 
  {props.children}
  </Stack>
}

const AuthenticationPage = function () {
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const location = useLocation()
  const from = (location?.state as { from?: Location })?.from?.pathname || '/'
  const handleLoginSubmit: React.FormEventHandler = async (event) => {
    event.preventDefault()
    const result = await dispatch(login({
      username: formValues.email,
      password: formValues.password
    }))
    if (result.payload) navigate(from, { replace: true })
    setFailedLogin(true)
  }
  const [formValues, setFormValues ] = useState({email: '', password: ''})
  const [ failedLogin, setFailedLogin ] = useState(false)
  const handleFormChange = (field: keyof typeof formValues): React.ChangeEventHandler<HTMLInputElement> => (event) => {
    event.preventDefault()
    setFormValues({
      ...formValues,
      [field]: event.target.value
    })
  }
  return (
    <form onSubmit={handleLoginSubmit}>
    <FormControl component={stack}>
      <LargerBoldText>Sign in</LargerBoldText>
      {failedLogin && <Alert severity="error">Failed to login. Incorrect email or password.</Alert>}
      <TextField label="Email" onChange={handleFormChange('email')}/>
      <TextField type="password" label="Password" onChange={handleFormChange('password')}/>
      <Button type="submit">Sign In</Button>
      <Text>Don't have an account? <Link to="/signup">Sign Up</Link></Text>
    </FormControl>
    </form>
  )
}

export default AuthenticationPage

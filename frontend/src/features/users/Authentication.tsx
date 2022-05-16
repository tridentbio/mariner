import React, { useState} from 'react'
import {Link} from 'react-router-dom'
import { Stack, TextField, FormControl, Button } from '@mui/material'
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
  const handleLoginSubmit: React.FormEventHandler = async (event) => {
    event.preventDefault()
    const result = await dispatch(login({
      username: formValues.email,
      password: formValues.password
    }))
  }
  const [formValues, setFormValues ] = useState({email: '', password: ''})
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
      <TextField label="Email" onChange={handleFormChange('email')}/>
      <TextField type="password" label="Password" onChange={handleFormChange('password')}/>
      <Button type="submit">Sign In</Button>
      <Text>Don't have an account? <Link to="/signup">Sign Up</Link></Text>
    </FormControl>
    </form>
  )
}

export default AuthenticationPage

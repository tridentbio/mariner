import React from 'react'
import {Link} from 'react-router-dom'
import { Stack, TextField, FormControl,FormControlLabel, InputLabel, Input, Button, Link as MLink } from '@mui/material'
import {Text, LargerBoldText} from '../../components/Text'

const stack =  (props: { children: React.ReactNode }) => {
  return <Stack display="flex" flexDirection="column" height={500} justifyContent="space-around" maxWidth={599} marginLeft="auto" marginRight="auto"> 
  {props.children}
  </Stack>
}

const AuthenticationPage = function () {
  return (
  <FormControl component={stack}>
    <LargerBoldText>Sign in</LargerBoldText>
      <TextField label="Email"/>
      <TextField label="Password"/>
      <Button>Sign In</Button>
    <Text>Don't have an account? <Link to="/login">Sign Up</Link></Text>
    </FormControl>
  )
}

export default AuthenticationPage

import React from 'react'
import { Typography } from '@mui/material'
import { SystemProps, TypographyProps } from '@mui/system'

export const Text: React.FC<TypographyProps & SystemProps & { children: React.ReactNode }> = (props) => {
  return <Typography {...props}> 
  {props?.children || null}
  </Typography>
}

export const LargerBoldText: React.FC<{children: React.ReactNode}> = (props) => {
  return <Text fontSize="h5.fontSize">{props.children}</Text>
}

import React, { useEffect } from 'react'
import {useNavigate, useLocation} from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../app/hooks'
import { TOKEN } from '../app/local-storage'
import {fetchMe} from '../features/users/usersSlice'
import {Text} from '../components/Text'

const RequireAuth: React.FC<{children:React.ReactNode}> = (props) => {
  const {loggedIn, fetchMeStatus} = useAppSelector(state => state.users)
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const location = useLocation()
  useEffect(() => {
    const token = localStorage.getItem(TOKEN)
    if (token && !loggedIn) {
      dispatch(fetchMe())
    }
  }, [])
  if (!loggedIn && ['loading', 'idle'].includes(fetchMeStatus)) {
    return <Text>Loading...</Text>
  } else if (!loggedIn) {
    navigate('/login', { replace: true, state: location })
  }
  // is logged in
  return <>{props.children}</>
}

export default RequireAuth

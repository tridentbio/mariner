import React, { useEffect } from 'react'
import {useNavigate, useLocation, Navigate} from 'react-router-dom'
import { useAppDispatch, useAppSelector } from '../app/hooks'
import { TOKEN } from '../app/local-storage'
import {fetchMe} from '../features/users/usersSlice'
import {Text} from '../components/Text'

const RequireAuth: React.FC<{children:React.ReactNode}> = (props) => {
  const {loggedIn, fetchMeStatus} = useAppSelector(state => state.users)
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const location = useLocation()
  const token = localStorage.getItem(TOKEN)
  const goLogin = () => 
    navigate('/login', { replace: true, state: {from: location} })
  const fetchUser = async () => {
      const result = await dispatch(fetchMe())
      return result.payload
  }
  useEffect(() => {
    if (token && fetchMeStatus !== 'loading' && !loggedIn) {
      fetchUser().then(user => !user && goLogin())
    } else if (!token) {
      goLogin()
    }
  }, [loggedIn, fetchMeStatus, token ])
  if (!token) {
    return <Navigate to="/login" state={{from: location}}/>
  } else if (['loading'].includes(fetchMeStatus)) {
    return <Text>Loading...</Text>
  } else {
    return <>{props.children}</>
  }
}

export default RequireAuth

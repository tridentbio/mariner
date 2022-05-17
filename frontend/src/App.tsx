import { Routes, Route, Outlet } from 'react-router-dom'
import AuthenticationPage from './features/users/Authentication'
import DatasetsPage from './features/datasets/Datasets'
import RequireAuth from './components/RequireAuth'
import { TopBar } from './components/TopBar'
import Dashboard from './features/dashboard/Dashboard'

function AppLayout () {
  return <>
    <TopBar/>
    <Outlet/>
  </>
}

function App () {
  return (
    <div className="App">
      <Routes>
        <Route path="/login" element={<AuthenticationPage/>}/>
        <Route element={<AppLayout/>}>
          <Route path="/datasets" element={<RequireAuth><DatasetsPage/></RequireAuth>}/>
          <Route path="/" element={<RequireAuth><Dashboard/></RequireAuth>}/>
        </Route>
      </Routes>
    </div>
  )
}

export default App

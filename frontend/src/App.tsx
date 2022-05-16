import React from 'react';
import {Routes, Route} from 'react-router-dom'
import AuthenticationPage from './features/users/Authentication';
import DatasetsPage from './features/datasets/Datasets'
import RequireAuth from './components/RequireAuth'

function App() {
  return (
    <div className="App">
      <Routes> 
        <Route path="/login" element={<AuthenticationPage/>}/>
        <Route>
          <Route path="/datasets" element={<RequireAuth><DatasetsPage/></RequireAuth>}/>
        </Route>
      </Routes>
    </div>
  );
}

export default App;

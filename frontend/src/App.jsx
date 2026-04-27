import { BrowserRouter, Routes, Route } from 'react-router-dom'

// Pages (we'll create these files next)
import Upload from './pages/Upload'
import Dashboard from './pages/Dashboard'
import ModelComparison from './pages/ModelComparison'
import AnomalyReport from './pages/AnomalyReport'
import History from './pages/History'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"          element={<Upload />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/compare"   element={<ModelComparison />} />
        <Route path="/anomalies" element={<AnomalyReport />} />
        <Route path="/history"   element={<History />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
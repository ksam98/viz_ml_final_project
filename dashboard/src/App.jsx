import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './Dashboard';
import ErrorDetailPage from './ErrorDetailPage';
import ImageDetail from './ImageDetail';
import ErrorEvolutionPage from './ErrorEvolutionPage';
import ImageEvolutionPage from './ImageEvolutionPage';

import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/error/:errorType" element={<ErrorDetailPage />} />
        <Route path="/image/:imageId/:errorType" element={<ImageDetail />} />
        <Route path="/error-evolution/:errorType" element={<ErrorEvolutionPage />} />
        <Route path="/image-evolution/:imageId" element={<ImageEvolutionPage />} />
      </Routes>
    </Router>
  );
}

export default App;

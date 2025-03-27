import React, { useState, useEffect } from 'react';
import { 
  predictHealthRisks, 
  analyzeSymptoms 
} from './services/api';
import HealthAssessment from './components/HealthAssessment';
import RiskVisualization from './components/RiskVisualization';
import RecommendationPanel from './components/RecommendationPanel';

function App() {
  const [userData, setUserData] = useState({
    age: '',
    bmi: '',
    bloodPressure: '',
    cholesterol: '',
    glucose: '',
    smoking: false,
    alcoholConsumption: '',
    physicalActivity: ''
  });

  const [healthAnalysis, setHealthAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setUserData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const submitHealthAssessment = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await predictHealthRisks(userData);
      setHealthAnalysis(response);
    } catch (err) {
      setError('Failed to process health assessment');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="health-predictor-app">
      <h1>Personalized Health Risk Predictor</h1>
      
      <form onSubmit={submitHealthAssessment}>
        <div className="form-group">
          <label>Age</label>
          <input 
            type="number" 
            name="age"
            value={userData.age}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="form-group">
          <label>BMI</label>
          <input 
            type="number" 
            name="bmi"
            value={userData.bmi}
            onChange={handleInputChange}
            step="0.1"
            required
          />
        </div>

        {/* Add more input fields for other health parameters */}
        
        <button 
          type="submit" 
          disabled={isLoading}
        >
          {isLoading ? 'Analyzing...' : 'Get Health Assessment'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}

      {healthAnalysis && (
        <>
          <RiskVisualization data={healthAnalysis.risk_assessment} />
          <RecommendationPanel 
            recommendations={healthAnalysis.recommendations}
          />
        </>
      )}
    </div>
  );
}

export default App;

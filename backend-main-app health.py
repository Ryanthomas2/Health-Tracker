import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.disease_predictor import DiseasePredictor
from models.risk_analyzer import RiskAnalyzer
from models.recommendation_engine import RecommendationEngine
from utils.data_preprocessor import DataPreprocessor
from utils.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize models and utilities
try:
    disease_predictor = DiseasePredictor()
    risk_analyzer = RiskAnalyzer()
    recommendation_engine = RecommendationEngine()
    data_preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
except Exception as e:
    logger.error(f"Model initialization error: {e}")
    raise

@app.route('/api/health/predict', methods=['POST'])
def predict_health_risks():
    """
    Endpoint for comprehensive health risk prediction
    """
    try:
        # Validate and preprocess input data
        user_data = request.json
        if not user_data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Preprocess and engineer features
        processed_data = data_preprocessor.preprocess(user_data)
        enhanced_features = feature_engineer.transform(processed_data)
        
        # Predict potential health risks
        disease_predictions = disease_predictor.predict(enhanced_features)
        
        # Analyze comprehensive health risks
        risk_assessment = risk_analyzer.assess_risks(enhanced_features)
        
        # Generate personalized recommendations
        recommendations = recommendation_engine.generate(
            disease_predictions, 
            risk_assessment
        )
        
        return jsonify({
            "disease_predictions": disease_predictions,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations
        })
    
    except Exception as e:
        logger.error(f"Health prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/health/symptoms', methods=['POST'])
def analyze_symptoms():
    """
    Endpoint for detailed symptom analysis
    """
    try:
        symptoms_data = request.json
        
        # Perform comprehensive symptom analysis
        symptom_analysis = disease_predictor.analyze_symptoms(symptoms_data)
        
        return jsonify(symptom_analysis)
    
    except Exception as e:
        logger.error(f"Symptom analysis error: {e}")
        return jsonify({"error": "Symptom analysis failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

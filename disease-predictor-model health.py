import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

class DiseasePredictor:
    def __init__(self, model_path='models/disease_predictor_model'):
        """
        Initialize the Disease Prediction Model
        """
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        
        # Load and preprocess training data
        self._load_training_data()
        
        # Attempt to load pre-trained weights if exists
        try:
            self.model.load_weights(model_path)
            print("Loaded pre-trained model weights")
        except:
            print("Training new model")
            self._train_model()
    
    def _load_training_data(self):
        """
        Load and preprocess health dataset
        """
        try:
            # Load comprehensive health dataset
            df = pd.read_csv('data/health_dataset.csv')
            
            # Preprocess features
            features = [
                'age', 'bmi', 'blood_pressure', 'cholesterol', 
                'glucose', 'smoking', 'alcohol_consumption',
                'physical_activity'
            ]
            X = df[features]
            
            # Multi-label disease classification
            y = df['diseases']
            
            # One-hot encode target diseases
            y_encoded = self.mlb.fit_transform(y.str.split(','))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
        
        except Exception as e:
            print(f"Error loading training data: {e}")
            raise
    
    def _build_model(self):
        """
        Build neural network for disease prediction
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(
                len(self.mlb.classes_), 
                activation='sigmoid'
            )
        ])
        
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
        return model
    
    def _train_model(self, epochs=100):
        """
        Train the disease prediction model
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        print(f"Test Accuracy: {test_accuracy}")
        
        # Save model weights
        self.model.save_weights('models/disease_predictor_model')
    
    def predict(self, user_data):
        """
        Predict potential diseases for user
        
        :param user_data: Processed user health features
        :return: Disease predictions with probabilities
        """
        try:
            # Scale input features
            scaled_features = self.scaler.transform(user_data)
            
            # Make multi-label prediction
            predictions = self.model.predict(scaled_features)
            
            # Convert predictions to interpretable results
            disease_probabilities = {}
            for i, disease in enumerate(self.mlb.classes_):
                disease_probabilities[disease] = float(predictions[0][i])
            
            return {
                "predictions": disease_probabilities,
                "high_risk_diseases": [
                    disease for disease, prob in disease_probabilities.items() 
                    if prob > 0.5
                ]
            }
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"error": "Unable to generate prediction"}
    
    def analyze_symptoms(self, symptoms):
        """
        Perform detailed symptom analysis
        
        :param symptoms: List of user-reported symptoms
        :return: Symptom analysis and potential conditions
        """
        try:
            # Load symptom database
            with open('data/symptom_database.json', 'r') as f:
                symptom_db = json.load(f)
            
            # Match symptoms to potential conditions
            potential_conditions = {}
            for symptom in symptoms:
                if symptom in symptom_db:
                    for condition, confidence in symptom_db[symptom].items():
                        potential_conditions[condition] = max(
                            potential_conditions.get(condition, 0), 
                            confidence
                        )
            
            # Rank and return top conditions
            sorted_conditions = sorted(
                potential_conditions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                "top_conditions": sorted_conditions[:5],
                "recommendation": "Consult a healthcare professional for accurate diagnosis"
            }
        
        except Exception as e:
            print(f"Symptom analysis error: {e}")
            return {"error": "Symptom analysis failed"}

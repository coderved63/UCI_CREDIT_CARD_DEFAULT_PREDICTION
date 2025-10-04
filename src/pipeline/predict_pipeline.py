import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and preprocessor once during initialization with validation"""
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            # Validate files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
            
            print("Loading model artifacts...")
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)
            self.model_loaded = True
            print(f"✅ Artifacts loaded: {self.model.__class__.__name__}")
            
        except Exception as e:
            print(f"❌ Failed to load artifacts: {e}")
            raise CustomException(e, sys)

    def predict(self, features):
        """Make binary predictions with validation"""
        try:
            if not self.model_loaded:
                raise ValueError("Model not properly loaded")
                
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, features):
        """Get prediction probabilities with fallback for models without proba"""
        try:
            if not self.model_loaded:
                raise ValueError("Model not properly loaded")
                
            data_scaled = self.preprocessor.transform(features)
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(data_scaled)
                return proba
            else:
                # For models without predict_proba, create confidence-based probabilities
                preds = self.model.predict(data_scaled)
                prob_array = []
                for pred in preds:
                    if pred == 1:
                        # Default prediction - moderate confidence
                        prob_array.append([0.25, 0.75])
                    else:
                        # No default prediction - moderate confidence  
                        prob_array.append([0.75, 0.25])
                return np.array(prob_array)
        except Exception as e:
            print(f"Probability prediction error: {e}")
            return None

    def get_model_info(self):
        """Get comprehensive information about the loaded model"""
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            model_info = {
                "model_name": self.model.__class__.__name__,
                "model_module": self.model.__module__,
                "has_predict_proba": hasattr(self.model, 'predict_proba'),
                "has_feature_importance": hasattr(self.model, 'feature_importances_'),
                "model_loaded": self.model_loaded
            }
            
            # Add feature information
            if hasattr(self.model, 'n_features_in_'):
                model_info["n_features"] = self.model.n_features_in_
            if hasattr(self.model, 'feature_names_in_'):
                model_info["feature_names"] = list(self.model.feature_names_in_)
            
            # Add model-specific parameters
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                model_info["key_parameters"] = self._extract_key_params(params)
            
            return model_info
            
        except Exception as e:
            return {"error": f"Failed to get model info: {str(e)}"}
    
    def _extract_key_params(self, params):
        """Extract key parameters based on model type"""
        model_name = self.model.__class__.__name__
        
        key_params = {}
        if 'RandomForest' in model_name or 'ExtraTrees' in model_name:
            for param in ['n_estimators', 'max_depth', 'min_samples_split']:
                if param in params:
                    key_params[param] = params[param]
        elif 'XGB' in model_name:
            for param in ['n_estimators', 'max_depth', 'learning_rate']:
                if param in params:
                    key_params[param] = params[param]
        elif 'CatBoost' in model_name:
            for param in ['iterations', 'depth', 'learning_rate']:
                if param in params:
                    key_params[param] = params[param]
        elif 'Logistic' in model_name:
            for param in ['C', 'penalty', 'solver']:
                if param in params:
                    key_params[param] = params[param]
        
        return key_params
    
    def validate_input(self, features_df):
        """Validate input features match model expectations"""
        try:
            if not self.model_loaded:
                return False, "Model not loaded"
            
            expected_features = getattr(self.model, 'n_features_in_', None)
            if expected_features and len(features_df.columns) != expected_features:
                return False, f"Expected {expected_features} features, got {len(features_df.columns)}"
            
            return True, "Validation passed"
        except Exception as e:
            return False, str(e)



class CustomData:
    def __init__(self,
        LIMIT_BAL: float,
        SEX: int,
        EDUCATION: int,
        MARRIAGE: int,
        AGE: int,
        PAY_0: int,
        PAY_2: int,
        PAY_3: int,
        PAY_4: int,
        PAY_5: int,
        PAY_6: int,
        BILL_AMT1: float,
        BILL_AMT2: float,
        BILL_AMT3: float,
        BILL_AMT4: float,
        BILL_AMT5: float,
        BILL_AMT6: float,
        PAY_AMT1: float,
        PAY_AMT2: float,
        PAY_AMT3: float,
        PAY_AMT4: float,
        PAY_AMT5: float,
        PAY_AMT6: float):

        self.LIMIT_BAL = LIMIT_BAL
        self.SEX = SEX
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.AGE = AGE
        self.PAY_0 = PAY_0
        self.PAY_2 = PAY_2
        self.PAY_3 = PAY_3
        self.PAY_4 = PAY_4
        self.PAY_5 = PAY_5
        self.PAY_6 = PAY_6
        self.BILL_AMT1 = BILL_AMT1
        self.BILL_AMT2 = BILL_AMT2
        self.BILL_AMT3 = BILL_AMT3
        self.BILL_AMT4 = BILL_AMT4
        self.BILL_AMT5 = BILL_AMT5
        self.BILL_AMT6 = BILL_AMT6
        self.PAY_AMT1 = PAY_AMT1
        self.PAY_AMT2 = PAY_AMT2
        self.PAY_AMT3 = PAY_AMT3
        self.PAY_AMT4 = PAY_AMT4
        self.PAY_AMT5 = PAY_AMT5
        self.PAY_AMT6 = PAY_AMT6

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "LIMIT_BAL": [self.LIMIT_BAL],
                "SEX": [self.SEX],
                "EDUCATION": [self.EDUCATION],
                "MARRIAGE": [self.MARRIAGE],
                "AGE": [self.AGE],
                "PAY_0": [self.PAY_0],
                "PAY_2": [self.PAY_2],
                "PAY_3": [self.PAY_3],
                "PAY_4": [self.PAY_4],
                "PAY_5": [self.PAY_5],
                "PAY_6": [self.PAY_6],
                "BILL_AMT1": [self.BILL_AMT1],
                "BILL_AMT2": [self.BILL_AMT2],
                "BILL_AMT3": [self.BILL_AMT3],
                "BILL_AMT4": [self.BILL_AMT4],
                "BILL_AMT5": [self.BILL_AMT5],
                "BILL_AMT6": [self.BILL_AMT6],
                "PAY_AMT1": [self.PAY_AMT1],
                "PAY_AMT2": [self.PAY_AMT2],
                "PAY_AMT3": [self.PAY_AMT3],
                "PAY_AMT4": [self.PAY_AMT4],
                "PAY_AMT5": [self.PAY_AMT5],
                "PAY_AMT6": [self.PAY_AMT6],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

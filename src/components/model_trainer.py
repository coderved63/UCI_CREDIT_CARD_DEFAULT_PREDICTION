import os 
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
# import lightgbm as lgb  # Optional - install with: pip install lightgbm

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # Optimized models for faster training (2-5 minutes instead of 17+ minutes)
            models = {
                "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1),
                "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=50),
                "CatBoost": CatBoostClassifier(random_state=42, verbose=False, iterations=50),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
                "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=50),
                # "SVM": SVC(random_state=42),  # Commented out - too slow for large datasets
                "Extra Trees": ExtraTreesClassifier(random_state=42, n_estimators=50, n_jobs=-1)
            }
            
            print(f"🚀 Training {len(models)} models with GridSearchCV optimization...")
            print(f"⚡ Using exhaustive parameter search with parallel processing for optimal results")

            # GridSearchCV optimized hyperparameter grids
            params={
                "Random Forest":{
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15],
                    'min_samples_split': [2, 5]
                },
                "XGBoost":{
                    'n_estimators': [50, 100],
                    'max_depth': [3, 6],
                    'learning_rate': [0.1, 0.2]
                },
                "Gradient Boosting":{
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                },
                "CatBoost":{
                    'iterations': [50, 100],
                    'depth': [4, 6],
                    'learning_rate': [0.1, 0.2]
                },
                "Logistic Regression":{
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['liblinear']
                },
                "AdaBoost":{
                    'n_estimators': [50, 100],
                    'learning_rate': [0.5, 1.0]
                },
                "Extra Trees":{
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15],
                    'min_samples_split': [2, 5]
                }
            }
            
            print(f"🔧 GridSearchCV hyperparameter search for optimal model performance...")
            model_report, fitted_models = evaluate_models(X_train,y_train,X_test,y_test,models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = fitted_models[best_model_name]  # Use fitted model instead of original

            if best_model_score<0.6:
                raise CustomException("No good model found")
            logging.info("Best model found on both training and testing data set")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            y_test_prob = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate comprehensive classification metrics
            from sklearn.metrics import roc_auc_score
            accuracy = accuracy_score(y_test, predicted)
            roc_auc = roc_auc_score(y_test, y_test_prob)
            report = classification_report(y_test, predicted, output_dict=True)
            conf_matrix = confusion_matrix(y_test, predicted)
            
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Model ROC-AUC: {roc_auc}")
            logging.info(f"Best Model Accuracy: {accuracy}")
            logging.info(f"Classification Report:\n{classification_report(y_test, predicted)}")
            
            print(f"\n{'='*60}")
            print(f" BEST MODEL: {best_model_name}")
            print(f" ROC-AUC: {roc_auc:.4f} (Primary Metric)")
            print(f" ACCURACY: {accuracy:.4f} (Secondary)")
            print(f"PRECISION: {report['macro avg']['precision']:.4f}")
            print(f" RECALL: {report['macro avg']['recall']:.4f}")
            print(f" F1-SCORE: {report['macro avg']['f1-score']:.4f}")
            print(f"{'='*60}")
            print(f"💡 Check logs for detailed classification report and confusion matrix")
            
            return roc_auc  
            

        except Exception as e:
            raise CustomException(e,sys) 
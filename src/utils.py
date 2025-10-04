import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        fitted_models = {}  # Store fitted models
        import time
        
        print(f"🚀 Starting model evaluation with GridSearchCV hyperparameter tuning...")
        print(f"📊 Dataset: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"⚡ Using exhaustive GridSearchCV for optimal parameter combinations")
        
        total_start_time = time.time()

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]
            
            start_time = time.time()
            print(f"\n📈 [{i+1}/{len(models)}] Tuning {model_name}...", end=" ", flush=True)

            # GridSearchCV with ROC-AUC scoring for comprehensive parameter search
            if len(para) > 0:  # Only if parameters exist
                gs = GridSearchCV(  # Using GridSearchCV for exhaustive parameter search
                    model, 
                    para, 
                    cv=2,      # 2-fold CV for speed
                    scoring='roc_auc',  # 🎯 Using ROC-AUC for financial models
                    n_jobs=-1, # Use all CPU cores
                    verbose=0, 
                    error_score='raise'
                )
                gs.fit(X_train, y_train)
                
                # Use best model directly
                best_model = gs.best_estimator_
                elapsed_time = time.time() - start_time
                print(f"✅ ({elapsed_time:.1f}s) - Best params: {gs.best_params_}")
                
            else:
                # If no parameters, just fit the model directly
                best_model = model
                best_model.fit(X_train, y_train)
                elapsed_time = time.time() - start_time
                print(f"✅ ({elapsed_time:.1f}s) - No tuning needed")

            # Store the fitted model
            fitted_models[model_name] = best_model

            # Evaluate the best model with both accuracy and ROC-AUC
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            y_test_prob = best_model.predict_proba(X_test)[:, 1]

            from sklearn.metrics import roc_auc_score
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_roc_auc = roc_auc_score(y_train, y_train_prob)
            test_roc_auc = roc_auc_score(y_test, y_test_prob)

            report[model_name] = test_roc_auc  # 🎯 Store ROC-AUC as primary score
            
            # Show progress with both metrics
            print(f"   🎯 Train ROC-AUC: {train_roc_auc:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")
            print(f"   📊 Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        total_elapsed_time = time.time() - total_start_time
        print(f"\n⏱️  Total evaluation time: {total_elapsed_time:.1f} seconds ({total_elapsed_time/60:.1f} minutes)")
        print(f"🎯 GridSearchCV evaluation completed successfully!")
        print(f"✅ Exhaustive parameter search ensures optimal model performance")
        
        return report, fitted_models  # Return both scores and fitted models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
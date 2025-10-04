import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import os

class CreditCardFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for credit card feature engineering
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # 1. Data Cleaning - Fix invalid categorical codes (MUST be done first)
        if 'EDUCATION' in X_copy.columns:
            # EDUCATION: 0, 5, 6 are invalid codes -> map to 4 (Others)
            # Valid codes: 1=Graduate, 2=University, 3=High School, 4=Others
            X_copy['EDUCATION'] = X_copy['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
            
        if 'MARRIAGE' in X_copy.columns:
            # MARRIAGE: 0 is invalid code -> map to 3 (Others)  
            # Valid codes: 1=Married, 2=Single, 3=Others
            X_copy['MARRIAGE'] = X_copy['MARRIAGE'].replace({0: 3})
        
        # 2. Credit Utilization Features
        X_copy['credit_utilization'] = np.where(
            X_copy['LIMIT_BAL'] > 0,
            X_copy['BILL_AMT1'] / X_copy['LIMIT_BAL'],
            0
        )
        
        # 3. Payment Ratio Features (for last 6 months)
        for i in range(1, 7):
            bill_col = f'BILL_AMT{i}'
            pay_col = f'PAY_AMT{i}'
            X_copy[f'payment_ratio_{i}'] = np.where(
                X_copy[bill_col] > 0,
                X_copy[pay_col] / X_copy[bill_col],
                0
            )
        
        # 4. Payment behavior analysis (using correct column names)
        pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        available_pay_cols = [col for col in pay_cols if col in X_copy.columns]
        
        if available_pay_cols:
            # Convert to numeric for calculations, then we'll keep originals as categorical
            pay_values = X_copy[available_pay_cols].astype(float)
            X_copy['avg_payment_status'] = pay_values.mean(axis=1)
            X_copy['max_payment_delay'] = pay_values.max(axis=1)
            X_copy['consecutive_delays'] = (pay_values > 0).sum(axis=1)
            # Count how many months had delays
            X_copy['months_with_delays'] = (pay_values > 0).sum(axis=1)
            # Check if ever paid early
            X_copy['ever_paid_early'] = (pay_values < 0).any(axis=1).astype(int)
        
        # 5. Bill amount trends
        bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
        X_copy['avg_bill_amount'] = X_copy[bill_cols].mean(axis=1)
        X_copy['bill_amount_trend'] = X_copy['BILL_AMT1'] - X_copy['BILL_AMT6']
        
        # 6. Payment amount trends  
        payment_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
        X_copy['avg_payment_amount'] = X_copy[payment_cols].mean(axis=1)
        X_copy['payment_amount_trend'] = X_copy['PAY_AMT1'] - X_copy['PAY_AMT6']
        
        # 7. High risk indicators
        X_copy['high_utilization'] = (X_copy['credit_utilization'] > 0.8).astype(int)
        # Create payment reliability score (lower is better)
        if available_pay_cols:
            X_copy['payment_reliability_score'] = pay_values.mean(axis=1)  # Average delay
        
        return X_copy

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation for UCI Credit Card dataset
        
        '''
        try:
            # Define numerical columns for UCI Credit Card dataset (including ordinal PAY variables)
            numerical_columns = [
                'LIMIT_BAL', 'AGE',
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'  # Ordinal: payment delay in months
            ]
            
            # Define categorical columns for UCI Credit Card dataset (only truly categorical variables)
            categorical_columns = [
                'SEX', 'EDUCATION', 'MARRIAGE'  # Only demographic categories, not ordinal payment status
            ]
            
            # Engineered features (will be created by feature engineering step)
            engineered_numerical_columns = [
                'credit_utilization', 'avg_payment_status', 'max_payment_delay',
                'avg_bill_amount', 'bill_amount_trend', 'avg_payment_amount', 
                'payment_amount_trend', 'months_with_delays', 'payment_reliability_score'
            ] + [f'payment_ratio_{i}' for i in range(1, 7)]
            
            engineered_categorical_columns = ['high_utilization', 'ever_paid_early']

            # Create numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Create categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first', handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            # All numerical columns (original + engineered)
            all_numerical_columns = numerical_columns + engineered_numerical_columns
            all_categorical_columns = categorical_columns + engineered_categorical_columns

            logging.info(f"Original numerical columns: {numerical_columns}")
            logging.info(f"Original categorical columns: {categorical_columns}")
            logging.info(f"Engineered numerical columns: {engineered_numerical_columns}")
            logging.info(f"Engineered categorical columns: {engineered_categorical_columns}")

            # Create the preprocessor with feature engineering
            preprocessor = Pipeline(
                steps=[
                    ("feature_engineer", CreditCardFeatureEngineer()),
                    ("column_transformer", ColumnTransformer(
                        transformers=[
                            ("num_pipeline", num_pipeline, all_numerical_columns),
                            ("cat_pipeline", cat_pipeline, all_categorical_columns)
                        ],
                        remainder='drop'  # Drop any remaining columns
                    ))
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for data transformation for UCI Credit Card dataset
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            # Define target column for UCI Credit Card dataset
            target_column_name = "default.payment.next.month"
            
            # Check if target column exists, if not try alternative names
            if target_column_name not in train_df.columns:
                possible_targets = [col for col in train_df.columns if 'default' in col.lower()]
                if possible_targets:
                    target_column_name = possible_targets[0]
                    logging.info(f"Using target column: {target_column_name}")
                else:
                    raise ValueError("Target column not found in dataset")
            
            # Drop ID column if it exists
            columns_to_drop = ['ID'] if 'ID' in train_df.columns else []
            if columns_to_drop:
                train_df = train_df.drop(columns=columns_to_drop)
                test_df = test_df.drop(columns=columns_to_drop)
                logging.info(f"Dropped columns: {columns_to_drop}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Input features shape - Train: {input_feature_train_df.shape}, Test: {input_feature_test_df.shape}")
            logging.info(f"Target distribution in train set:\n{target_feature_train_df.value_counts()}")

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Apply preprocessing (includes feature engineering)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info(f"Transformed features shape - Train: {input_feature_train_arr.shape}, Test: {input_feature_test_arr.shape}")

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed successfully")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
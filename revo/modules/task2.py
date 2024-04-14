from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb

from ml_helpers import remove_outliers_iqr



class classificator_model(ABC):

    def ml_preprocessing(self,IQR_multiplier, knn_neighbors,
                         df_path:str='output/0_clean_arff/df_task2.csv',
                         )->pd.DataFrame:
        # Import cleaned data
        df=pd.read_csv('output/0_clean_arff/df_task2.csv')
        
        df=df.drop(['description_sector','Quarter'],axis=1)
        df=pd.get_dummies(df)
        df = df.astype(float)
        
        # Change the code_sector to work with certain ml functions below
        df['code_sector'] = (df['code_sector'] - 1).astype(int)
        
        X_cols=df[['code_sector']].columns
        Y_cols=df.drop(['code_sector'],axis=1).columns
        
        # Split dataset into train-test
        X = df.drop(columns=['code_sector'])
        y = df['code_sector']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = remove_outliers_iqr(X_train,IQR_multiplier)
        X_test = remove_outliers_iqr(X_test,IQR_multiplier)
        
        # Inpute missing data with interpolation as it is time series data
        X_train = X_train.interpolate(method='linear')
        X_test = X_test.interpolate(method='linear')
             
        # Inpute the remaining missing values (due to missing from the start or from the end of df)
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.fit_transform(X_test)


    @abstractmethod
    def train_and_return_accuracy(self,df:pd.DataFrame='output/0_clean_arff/df_task2.csv')->None:
        pass     
    

class RandomForest(classificator_model):
   
    def train_and_return_accuracy(self,df:pd.DataFrame='output/0_clean_arff/df_task2.csv')->None:
        pass   
    
    
class LightGBM(classificator_model):
    
    def train_and_return_accuracy(self,df:pd.DataFrame='output/0_clean_arff/df_task2.csv')->None:
        pass   


class XGBoost(classificator_model):  
    
    def train_and_return_accuracy(self,df:pd.DataFrame='output/0_clean_arff/df_task2.csv')->None:
        pass   









 
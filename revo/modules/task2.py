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
from typing import Tuple, List



class classificator_model(ABC):

    def ml_preprocessing(self,IQR_multiplier:float, knn_neighbors:int,
                         test_size_percentage:float,
                         df_path:str='output/0_clean_arff/df_task2.csv'
                         )->Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        # Import cleaned data
        df=pd.read_csv('output/0_clean_arff/df_task2.csv')
        
        df=df.drop(['description_sector','Quarter'],axis=1)
        df=pd.get_dummies(df)
        df = df.astype(float)
        
        # Change the code_sector to work with certain ml functions below
        df['code_sector'] = (df['code_sector'] - 1).astype(int)
        
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
        
        # For more comprehensible output
        labels_list=['Transportation and warehousing','Wholesale trade','Manufacturing','Retail trade','Energy','Construction']
        
        return X_train,X_test,y_train,y_test,labels_list


    @abstractmethod
    def train_and_return_accuracy(self,df:pd.DataFrame='output/0_clean_arff/df_task2.csv')->None:
        pass     
    

class RandomForest(classificator_model):
   
    def train_and_return_accuracy(self,X_train:pd.DataFrame,X_test:pd.DataFrame,
                                  y_train:pd.Series,y_test:pd.Series,
                                  labels_list:List[str])->None:
        
        # Initialize and train the Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Classification report
        print(classification_report(y_test, y_pred,target_names=labels_list))
  
    
    
class LightGBM(classificator_model):
    
    def train_and_return_accuracy(self,X_train:pd.DataFrame,X_test:pd.DataFrame,
                                  y_train:pd.Series,y_test:pd.Series,
                                  labels_list:List[str])->None:
        
        # Initialize and train the LightGBM Classifier
        clf = lgb.LGBMClassifier(num_class=6, random_state=42,force_col_wise=True)
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Classification report
        print(classification_report(y_test, y_pred,target_names=labels_list))

   


class XGBoost(classificator_model):  
    
    def train_and_return_accuracy(self,X_train:pd.DataFrame,X_test:pd.DataFrame,
                                  y_train:pd.Series,y_test:pd.Series,
                                  labels_list:List[str])->None:
        
        # Initialize and train the XGBoost Classifier
        clf = xgb.XGBClassifier(objective='multi:softmax', num_class=6, random_state=42)
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Classification report
        print(classification_report(y_test, y_pred,target_names=labels_list))
   









 
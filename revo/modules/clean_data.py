import pandas as pd
import arff
import os 
import re
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from typing import Tuple

from ml_helpers import remove_outliers_iqr


class Dataset():
    '''Class that contains all the logic for importing and cleaning data.
    The inputs are .arff and csv; the outputs are csv.'''
    
    def __init__(self,data_path:str='data/',
                      dimension_path:str='dimension/dimension.csv',
                      sector_path:str='dimension/sector_dimension.csv',
                      data_modified_path:str='data_modified/')->None:
        
        self.data_path=data_path
        self.dimension_path=dimension_path 
        self.sector_path=sector_path
        self.data_modified_path=data_modified_path
        
    def import_metadata(self)->Tuple[pd.DataFrame, pd.DataFrame]:
        dimension=pd.read_csv(self.dimension_path,delimiter=';')
        sector=pd.read_csv(self.sector_path)
        return dimension,sector

    
    def import_clean_output_arff(self)->None:
        '''
        - Imports .arff files from data_path
        - Clean the .arff files:
            Step 1: Clean random letters appearing and convert them to m
            Step 2: Clean the letter m in the Sector column which should be numeric,
                  we know it`s the last column, convert m to ?
                  which is the null value for numeric attributes in .arff files
            Step 3: Remove rows where the country Italy appears inb the first column.
        - Outputs the cleaned .arff files in data_modified_path
        '''

        for filename in os.listdir(self.data_path):# Loop through each file in the directory
            with open(f'{self.data_path}{filename}', 'r') as rf: # Open the .arff file
                # Read lines from the file
                lines = rf.readlines()

            for i, line in enumerate(lines):
                
                # Step 1: 
                # Clean random letters appearing and convert them to m
                pattern = r',([a-z]),'# Regex to match comma + single lowercase letter + comma
                # Perform the substitution
                lines[i] = re.sub(pattern, lambda x: ',m,', line)
                
                # Step 2:
                # If there is an m in the last three characters
                if 'm' in line[-3:]:
                    # If yes, replace 'm' with '?' in the last 3 characters
                    lines[i] = line[:-3] + line[-3:].replace('m', '?')
            
                # Step 3:
                # Split the string by commas
                string_elements = line.split(',')
                # Get the first value
                first_value = string_elements[0]

                # Check if the first value is Italy
                if first_value == 'Italy':
                    # If not, mark the line with 'Remove'
                    lines[i] = 'Remove'
                    
            # Let`s remove all the lines=='Remove'
            lines = [line for line in lines if line != 'Remove']     
            
            # Write modified lines back to a new ARFF file
            with open(f'{self.data_modified_path}{filename}', 'w') as wf:
                wf.writelines(lines)

    def import_clean_arff_to_df(self,dimension:pd.DataFrame)->pd.DataFrame:
        '''Imports the cleaned up arff files from data_modified and puts them in a single df.'''
        
        # Initialize an empty list to store DataFrames
        dfs_list = []

        for filename in os.listdir(self.data_modified_path):# Loop through each file in the directory
            with open(f'{self.data_modified_path}{filename}', 'r') as f: # Open the .arff file
            
                raw_data = arff.load(f) # dict obj
                # Store arff in a pandas df
                df = pd.DataFrame(raw_data['data'],columns=[x[0] for x in raw_data['attributes']])
                
                # Create a dict with the mappings
                mappings = dict(zip(dimension['Variable Name'], dimension['Description']))
                # Rename columns in df using the mapping
                df.rename(columns=mappings,inplace=True)

                # Store year and quarter cols
                df['Year'] = int(filename[:4])
                df['Quarter'] = filename[5:7]
                
                # Append DataFrame to the list
                dfs_list.append(df)
                
        # Concatenate all DataFrames in the list
        df = pd.concat(dfs_list, ignore_index=True)
        
        return df
        
    def output_task1_and_task2_data(self,df,sector:pd.DataFrame,na_row_threshold:float=0.7,
                                    IQR_multiplier:float=4)->None:
        '''Cleaning step to the entire dataframe.
        na_row_threshold is the percentage where if a row has more than that many null values,
        it gets dropped.
        
        Outputs prepared data for task1 and task2.'''
        
        # Handle missing values
        df.replace('m', None, inplace=True)
        # Remove rows with no county
        df=df.dropna(subset=['Country']).reset_index(drop=True)
        # Remove rows with no sector
        df=df.dropna(subset=['sectors']).reset_index(drop=True)
        # Convert financial indicator cols into floats
        cols=df.drop(['Country','Year','Quarter','sectors'],axis=1).columns
        df[cols] = df[cols].astype(float)
        # Remove rows where more than threshold percent of cols are null
        threshold = int(na_row_threshold * len(df.columns))
        df=df.dropna(thresh=threshold).reset_index(drop=True)
        
        # Get the sector mapping in, we choose inner to get rid of a sector=0 row
        df=df.merge(sector,how='inner', left_on='sectors', right_on='code_sector').drop(['sectors'],axis=1)
        
        # Extract numeric part from 'Quarter' column and convert to float
        df['Quarter'] = df['Quarter'].str.extract('(\d+)').astype(float)

        # Sort df by year and quarter
        df=df.sort_values(['Year','Quarter'])

        # Convert numeric cols to float
        df = pd.concat([df[['Country','description_sector']], 
                        df.drop(['Country','description_sector'],
                                axis=1).astype(float)], axis=1)
        
        # Encode Quarter to preserve its cyclic nature (useful for ml tasks)

        # Define the period for the trigonometric encoding (4 for quarters in a year)
        period = 4
        # Apply trigonometric coding
        df['sin_quarter'] = np.sin(2 * np.pi * df['Quarter'] / period)
        df['cos_quarter'] = np.cos(2 * np.pi * df['Quarter'] / period)

        # Create missing indicator columns
        for col in cols:
            missing_indicator_col_name = 'MI_'+col
            df[missing_indicator_col_name] = df[col].isnull().astype(int)
            
        # To prevent leakage, we will output the data for task2 before any outlier and 
        # missing data inputing technique
        df.to_csv('output/0_clean_arff/df_task2.csv',index=False)
        print('\n Exported df_task2.csv in output/0_clean_arff')

        # Continuing for task1 data
        # I dont know if this makes financial sense but ,I will remove massive outliers because they seem off           
        df = remove_outliers_iqr(df,IQR_multiplier)
        
        df = df.copy() # Defragment dataframe
        df=df.sort_values(['Year','Quarter']).reset_index(drop=True)
        # Inpute missing data with interpolation as it is time series data
        df[cols] = df[cols].interpolate(method='linear')
        # Inpute the remaining missing values (due to missing from the start or from the end of df)
        imputer = KNNImputer(n_neighbors=5)
        df[cols] = imputer.fit_transform(df[cols])
        
        # As we are going to use linear regression, let s normalize the data
        scaler = StandardScaler()
        # Normalize
        df[cols] = scaler.fit_transform(df[cols])
        
        # Save the df_task1
        df.to_csv('output/0_clean_arff/df_task1.csv',index=False)
        print('\n Exported df_task1.csv in output/0_clean_arff')
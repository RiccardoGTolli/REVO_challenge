import argparse
import json
import pandas as pd

def select_config_file():
    parser = argparse.ArgumentParser(description="Select the config file and config set.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.json",
        help="Pass a config file like: config.json",
    )
    args = parser.parse_args()

    return args


def get_config_from_file(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def remove_outliers_iqr(df,IQR_multiplier):
    """
    Remove outliers from each column of a DataFrame using the Interquartile Range (IQR) method.
    Turns the outliers to None.
    Smaller values of IQR_multiplier will remove outliers closer to the mean,
    larger values of IQR_multiplier will only remove outliers farther from the mean.
    A general rule of thumb is IQR_multiplier=1.5
    """
    df_no_outliers = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    
    # Select columns with float dtype
    float_cols = df.select_dtypes(include=['float']).columns

    # Iterate over each float column
    for column in df[float_cols].columns:
        # Calculate the first quartile (Q1) and third quartile (Q3) for the column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        
        # Calculate the IQR for the column
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers for the column
        lower_bound = Q1 - IQR_multiplier * IQR
        upper_bound = Q3 + IQR_multiplier * IQR
        
        # Replace outliers with None for the column
        df_no_outliers[column] = df[column].mask((df[column] < lower_bound) | (df[column] > upper_bound), other=None)
    
    return df_no_outliers

def get_year_quarter_combos(start_year,start_quarter,
                            end_year,end_quarter):
    ''' Will return a dataframe that can be used to inner join df
    so that you can filter the df based on year and quarter.
    '''
    all_combos=[]
    for year in range(start_year,end_year+1):
        if year==start_year:
            for quarter in range(start_quarter,5):
                combo=(year,quarter)
                all_combos.append(combo)
        elif year!=start_year and year<end_year:
            for quarter in range(1,5):
                combo=(year,quarter)
                all_combos.append(combo)
        elif year==end_year:
            for quarter in range(1,end_quarter+1):
                combo=(year,quarter)
                all_combos.append(combo)
                
    # Convert the list of tuples to a DataFrame
    filter_df = pd.DataFrame(all_combos, columns=['Year', 'Quarter'])
    
    return filter_df
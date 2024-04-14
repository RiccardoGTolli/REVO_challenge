import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Dict, Union

def select_config_file()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the config file and config set.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.json",
        help="Pass a config file like: config.json",
    )
    args = parser.parse_args()
    
    return args

def get_config_from_file(config_file)->Dict[str, Union[int, float]]:
    with open(config_file, "r") as file:
        config = json.load(file)

    return config


def remove_outliers_iqr(df:pd.DataFrame,IQR_multiplier:float=3)->pd.DataFrame:
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

def get_year_quarter_combos(start_year:int,start_quarter:int,
                            end_year:int,end_quarter:int)->pd.DataFrame:
    ''' Will return a dataframe that can be used to inner join df
    so that you can filter the df based on year and quarter.
    '''
    
    # Assert statements to check if values are integers and positive
    assert isinstance(start_year, int) and start_year > 0, "start_year must be a positive integer"
    assert isinstance(start_quarter, int) and start_quarter > 0, "start_quarter must be a positive integer"
    assert isinstance(end_year, int) and end_year > 0, "end_year must be a positive integer"
    assert isinstance(end_quarter, int) and end_quarter > 0, "end_quarter must be a positive integer"
    
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


def plot_lin_reg_scatter(df:pd.DataFrame,col:str,
                         coeff:float,pvalue:float,dir:str)->None:
    '''Plot for linear regression model for task1_a and task1_b'''
    
    # Plot the points from y variables
    plt.scatter(df['Year'] + (df['Quarter'] - 1) / 4 ,
                df[col],
                color='blue')

    plt.xlabel('Year and Quarter')
    plt.ylabel(col)
    plt.title(col)
    
    # Plot the line of best fit on the fly (red)
    x_values = (df['Year'] + (df['Quarter'] - 1) / 4).astype('float') 
    y_values = df[f'Predicted {col}'].astype('float')     
    # Fit a polynomial of degree 1 (a straight line) to the data
    coefficients = np.polyfit(x_values, y_values, 1)    
    # Create a polynomial function based on the coefficients
    poly_function = np.poly1d(coefficients)    
    # Generate the x values for the line of best fit
    x_fit = np.linspace(min(x_values), max(x_values), 100)    
    # Calculate the corresponding y values using the polynomial function
    y_fit = poly_function(x_fit)    
    # Plot the line of best fit
    plt.plot(x_fit, y_fit, color='red', linestyle='-', label='Predicted Line')

    # Set formatter to suppress scientific notation on the x-axis
    plt.ticklabel_format(useOffset=False, style='plain')
    
    # Add a text box with coefficient and p-value
    text = f'Coeff: {coeff:.2f}\nP-value: {pvalue:.2f}'
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    # Save the plot to a file
    col_no_slashes=col.replace("/", "_") # Remove the slashes
    plt.savefig(f'output/{dir}/{col_no_slashes}.png')
    # Reset the current figure
    plt.clf()
    
    print(f'\nStatistically significant plot saved in output/{dir}/{col_no_slashes}.png')
    
    
def delete_files_in_folder_recursively(folder_path:str)->None:
    # Walk through the directory structure
    for root, dirs, files in os.walk(folder_path):
        # Iterate over the files in the directory
        for file in files:
            file_path = os.path.join(root, file)
            # Delete the file
            os.remove(file_path)
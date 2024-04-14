import argparse
import json

def select_config_file_set():
    parser = argparse.ArgumentParser(description="Select the config file and config set.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="train_config_1.json",
        help="Values are like: train_config_1.json",
    )
    parser.add_argument(
        "--config-set",
        type=str,
        default="NGB_1",
        help="Values are like: Values are like: NGB_1.",
    )
    args = parser.parse_args()

    return args


def get_config_from_file(config_file, config_set):
    with open(config_file, "r") as file:
        all_configs = json.load(file)
        config = all_configs[config_set]
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
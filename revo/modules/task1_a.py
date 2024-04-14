import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def task1_a():
    # Import cleaned data
    df=pd.read_csv('output/0. cleaned_data/df_task1.csv')
        


filter_df=get_year_quarter_combos(2019,4,2022,2)

# Filter df based on the provided start and end year and quarter
df=df.merge(filter_df,on=['Year','Quarter'])

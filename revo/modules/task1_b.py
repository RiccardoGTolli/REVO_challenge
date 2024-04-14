import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

from ml_helpers import get_year_quarter_combos,plot_lin_reg_scatter

def task1_b(start_year:int,start_quarter:int,
            end_year:int,end_quarter:int,
            pvalue:float=0.05):
    # Import cleaned data
    df=pd.read_csv('output/0_clean_arff/df_task1.csv')
    # Remove rows without sector
    df.dropna(subset='description_sector',inplace=True)
    
    # Filter df based on the provided start and end year and quarter
    filter_df=get_year_quarter_combos(start_year,start_quarter,end_year,end_quarter)
    df=df.merge(filter_df,on=['Year','Quarter'])
    
    df.drop(['code_sector','Country'],axis=1,inplace=True)
    
    X_cols=df[['Year','sin_quarter','cos_quarter']].columns		
    Y_cols=df.drop(['description_sector','Year','sin_quarter','cos_quarter','Quarter'],axis=1).columns
    
    # Get a df for each sector, same analysis as task1_a
    all_dfs_list=[]

    for sector in df['description_sector'].unique():
        sector_df=df[df['description_sector']==sector].copy()
        
        # Store the statistically significant columns in a df
        cols_list=[]
        coefficients_list=[]
        pvalues_list=[]

        # Fit a multiple linear regression model to each financial col
        for financial_col in Y_cols: 
            # Add a constant term and combine the 3 x variables into one array
            X = sm.add_constant(list(zip(sector_df['Year'], sector_df['sin_quarter'], sector_df['cos_quarter'])))  
            model = sm.OLS(sector_df[financial_col], X).fit()

            # Get model characteristics
            coefficients = model.params[1:] 
            p_values = model.pvalues[1:]
            
            # Save the column if the pvalue is significant
            if p_values.mean()<=0.05:
                # Predicted values from the model
                predicted_values = model.predict(X)

                # Add predictions to sector_df and df
                sector_df[f'Predicted {financial_col}']=predicted_values
                df[f'Predicted {financial_col}']=None # Initialize column
                df.loc[sector_df.index,f'Predicted {financial_col}']=predicted_values
                
                cols_list.append(financial_col)
                coefficients_list.append(coefficients.mean())
                pvalues_list.append(p_values.mean())
        
        sector_df_results = pd.DataFrame({'Statistical Significant Financial Indicator':cols_list,
                                    'p-value':pvalues_list,
                                    'slope':coefficients_list,
                                    'description_sector':sector}).sort_values(['slope'],
                                                                            ascending=False).reset_index(drop=True)
        all_dfs_list.append(sector_df_results)
        
    all_sectors_df=pd.concat(all_dfs_list).reset_index(drop=True)
    
    task1_b_answer=all_sectors_df['Statistical Significant Financial Indicator'].value_counts()
    
    task1_b_answer=pd.DataFrame({task1_b_answer.index.name:task1_b_answer.index.values,
                             'Rank':task1_b_answer}).reset_index(drop=True)
    
    # Save the task1_a_answer
    task1_b_answer.to_csv('output/3_task1_b/df_task1_answer.csv',index=False)
    # The plots for each column are in revo/output/task1_b_answers
    
    # Produce plots
    for financial_col,sector in zip(all_sectors_df['Statistical Significant Financial Indicator'],
                                    all_sectors_df['description_sector']):
        
        filtered_df=df[df['description_sector']==sector].copy()
        filtered_df=filtered_df[[financial_col,f'Predicted {financial_col}',
                                'Year','Quarter']]
        filtered_df['description_sector']=sector
        
        coeff=all_sectors_df[
        (all_sectors_df['description_sector']==sector) &
        (all_sectors_df['Statistical Significant Financial Indicator']==financial_col)
        ]['slope'].iloc[0]

        pvalue=all_sectors_df[
        (all_sectors_df['description_sector']==sector) &
        (all_sectors_df['Statistical Significant Financial Indicator']==financial_col)
        ]['p-value'].iloc[0]
        
        plot_lin_reg_scatter(filtered_df,financial_col,coeff,pvalue,'3_task1_b')
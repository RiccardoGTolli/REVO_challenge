import pandas as pd

import statsmodels.api as sm


from ml_helpers import get_year_quarter_combos,plot_lin_reg_scatter

def task1_a(start_year:int,start_quarter:int,
            end_year:int,end_quarter:int,
            pvalue:float=0.05):
    # Import cleaned data
    df=pd.read_csv('output/0_clean_arff/df_task1.csv')
    
    # Filter df based on the provided start and end year and quarter
    filter_df=get_year_quarter_combos(start_year,start_quarter,end_year,end_quarter)
    df=df.merge(filter_df,on=['Year','Quarter'])
    
    df.drop(['code_sector','description_sector','Country'],axis=1,inplace=True)
    
    X_cols=df[['Year','sin_quarter','cos_quarter']].columns		
    Y_cols=df.drop(['Year','sin_quarter','cos_quarter','Quarter'],axis=1).columns
    
    
    # Store the statistically significant columns in a df
    cols_list=[]
    coefficients_list=[]
    pvalues_list=[]

    # Fit a multiple linear regression model to each financial col
    for financial_col in Y_cols: 
        # Add a constant term and combine the 3 x variables into one array
        X = sm.add_constant(list(zip(df['Year'], df['sin_quarter'], df['cos_quarter'])))  
        model = sm.OLS(df[financial_col], X).fit()
    
        # Get model characteristics
        coefficients = model.params[1:] 
        p_values = model.pvalues[1:]
        
        # Save the column if the pvalue is significant
        if p_values.mean()<=0.05:
            # Predicted values from the model
            predicted_values = model.predict(X)
            # Add predictions to df
            df[f'Predicted {financial_col}']=predicted_values
            
            cols_list.append(financial_col)
            coefficients_list.append(coefficients.mean())
            pvalues_list.append(p_values.mean())
            
            # Save all the plots of statistical significant cols in filesystem
            plot_lin_reg_scatter(df,financial_col,coefficients.mean(),
                                 p_values.mean(),'2_task1_a')

    task1_a_answer = pd.DataFrame({'Statistical Significant Financial Indicator':cols_list,
                                   'p-value':pvalues_list,
                                   'slope':coefficients_list}).sort_values(['slope'],ascending=False
                                                                           ).reset_index(drop=True)
    # Save the task1_a_answer
    task1_a_answer.to_csv('output/2_task1_a/df_task1_answer.csv',index=False)
    # The plots for each column are in revo/output/task1_a_answers 
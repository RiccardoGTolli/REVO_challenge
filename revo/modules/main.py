from clean_data import Dataset
from ml_helpers import select_config_file,get_config_from_file,delete_files_in_folder_recursively
from task1_exploratory_analysis import task1_exploratory_analysis
from task1_a import task1_a
from task1_b import task1_b


# Retrieve config
config_argument = select_config_file() # Get the argument from the console 
config = get_config_from_file(config_argument.config_file)

# Delete any file in the output folder (from previous run)
if config['reset_output'] is True:
    delete_files_in_folder_recursively('output')
    
# Get cleaned data
print('ClEANING DATA...')
dataset=Dataset()
dimension,sector=dataset.import_metadata()
dataset.import_clean_output_arff()
df=dataset.import_clean_arff_to_df(dimension)
dataset.output_task1_and_task2_data(df,sector,config['knn_neighbors'],
                                    config['na_row_threshold'],
                                    config['IQR_multiplier'])

# Task 1_a
print('STARTING TASK1 EXPLORATORY ANALYSIS...')
task1_exploratory_analysis() # Plots saved in 'output/1_task1_exploratory_analysis/
print('STARTING TASK1 A...')
task1_a(config['start_year'],config['start_quarter'], # plots and df saved in 'output/2_task1_a/
        config['end_year'],config['end_quarter'],config['pvalue'])
# Task 1_b
print('STARTING TASK1 B...')
task1_b(config['start_year'],config['start_quarter'], # plots and df saved in 'output/3_task1_b/
        config['end_year'],config['end_quarter'],config['pvalue'])
# Task 2
print('STARTING TASK2...')


print('RUN COMPLETED. Check output/ for results and plots.')
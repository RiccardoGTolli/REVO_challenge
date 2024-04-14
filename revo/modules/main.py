from clean_data import Dataset
from ml_helpers import select_config_file,get_config_from_file,delete_files_in_folder_recursively
from task1_exploratory_analysis import task1_exploratory_analysis
from task1_a import task1_a


# Retrieve config
config_argument = select_config_file() # Get the argument from the console 
config = get_config_from_file(config_argument.config_file)

# Delete any file in the output folder (from previous run)
if config['reset_output'] is True:
    delete_files_in_folder_recursively('output')
    
# Get cleaned data
dataset=Dataset()
dimension,sector=dataset.import_metadata()
dataset.import_clean_output_arff()
df=dataset.import_clean_arff_to_df(dimension)
dataset.output_task1_and_task2_data(df,sector,config['na_row_threshold'],
                                    config['IQR_multiplier'])

# Task 1_a
task1_exploratory_analysis() # Plots saved in 'output/1_task1_exploratory_analysis/
task1_a(config['start_year'],config['start_quarter'],
        config['end_year'],config['end_quarter'],config['pvalue'])
# Tasl 1_b

# Task 2
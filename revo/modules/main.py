from clean_data import Dataset
from ml_helpers import select_config_file,get_config_from_file
from task1_exploratory_analysis import task1_exploratory_analysis


# Retrieve config
config_argument = select_config_file() # Get the argument from the console 
config = get_config_from_file(config_argument.config_file)

breakpoint()
# Get cleaned data
dataset=Dataset()
dimension,sector=dataset.import_metadata
dataset.import_clean_output_arff()
df=dataset.import_clean_arff_to_df()
dataset.output_task1_and_task2_data(df,config['na_row_threshold'],config['IQR_multiplier'])

# Task 1_a
task1_exploratory_analysis() # Plots saved in 'output/1. task1_exploratory_analysis/
# Tasl 1_b

# Task 2
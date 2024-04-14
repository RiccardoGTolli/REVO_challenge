from clean_data import Dataset


# Get cleaned data
dataset=Dataset()
dimension,sector=dataset.import_metadata
dataset.import_clean_output_arff()
df=dataset.import_clean_arff_to_df()
dataset.output_task1_and_task2_data(df,config['na_row_threshold'],config['IQR_multiplier'])

# Task 1_a

# Tasl 1_b

# Task 2
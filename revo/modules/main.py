from io import StringIO

import pandas as pd
from clean_data import Dataset
from ml_helpers import (
    delete_files_in_folder_recursively,
    get_config_from_file,
    select_config_file,
)
from task1_a import task1_a
from task1_b import task1_b
from task1_exploratory_analysis import task1_exploratory_analysis
from task2 import LightGBM_, RandomForest_, XGBoost_


def main():
    # Retrieve config
    config_argument = select_config_file()  # Get the argument from the console
    config = get_config_from_file(config_argument.config_file)

    # Delete any file in the output folder (from previous run)
    if config["reset_output"] is True:
        delete_files_in_folder_recursively("output")

    # Get cleaned data
    print("\nCLEANING DATA...")
    dataset = Dataset()
    dimension, sector = dataset.import_metadata()
    dataset.import_clean_output_arff()
    df = dataset.import_clean_arff_to_df(dimension)
    dataset.output_task1_and_task2_data(
        df, sector, config["knn_neighbors"], config["mi_cols"], config["na_row_threshold"], config["IQR_multiplier"]
    )

    # Task 1_a
    print("\nSTARTING TASK1 EXPLORATORY ANALYSIS...")
    task1_exploratory_analysis()  # Plots saved in 'output/1_task1_exploratory_analysis/
    print("\nSTARTING TASK1 A...")
    task1_a(
        config["start_year"],
        config["start_quarter"],  # plots and df saved in 'output/2_task1_a/
        config["end_year"],
        config["end_quarter"],
        config["pvalue"],
    )
    # Task 1_b
    print("\nSTARTING TASK1 B...")
    task1_b(
        config["start_year"],
        config["start_quarter"],  # plots and df saved in 'output/3_task1_b/
        config["end_year"],
        config["end_quarter"],
        config["pvalue"],
    )
    # Task 2
    print("\nSTARTING TASK2...")

    # RANDOM FOREST
    rf_ = RandomForest_()
    X_train, X_test, y_train, y_test, labels_list = rf_.ml_preprocessing(
        config["IQR_multiplier"],
        config["knn_neighbors"],
        config["test_size_percentage"],
        df_path="output/0_clean_arff/df_task2.csv",
    )
    rf_accuracy, rf_report = rf_.train_and_return_accuracy(
        X_train, X_test, y_train, y_test, labels_list, config["early_stopping"], config["rf_hyperparameters"]
    )

    # LIGHT GBM
    lgbm_ = LightGBM_()
    lgbm_accuracy, lgbm_report = lgbm_.train_and_return_accuracy(
        X_train, X_test, y_train, y_test, labels_list, config["early_stopping"], config["lgbm_hyperparameters"]
    )

    # XGBOOST
    xgb_ = XGBoost_()
    xgb_accuracy, xgb_report = xgb_.train_and_return_accuracy(
        X_train, X_test, y_train, y_test, labels_list, config["early_stopping"], config["xgboost_hyperparameters"]
    )

    # Create a dictionary to store variable names, their corresponding accuracies, and reports
    models_results = {
        "Random Forest": [rf_accuracy, rf_report],
        "LightGBM": [lgbm_accuracy, lgbm_report],
        "XGBoost": [xgb_accuracy, xgb_report],
    }
    # Find the maximum accuracy and the corresponding variable name and report
    best_model, (best_accuracy, best_report) = max(models_results.items(), key=lambda x: x[1][0])

    print(f"The best model is {best_model} with accuracy {best_accuracy}")

    # Convert the classification report to a DataFrame
    report_df = pd.read_fwf(StringIO(best_report), index_col=0)
    report_df["Total Model Accuracy"] = best_accuracy

    # Transforming a string to a df, so it`s not perfect
    report_df = report_df.replace("and warehousing", "Transportation and warehousing")

    # Save the DataFrame to a CSV file
    report_df.to_csv(f"output/4_task2/{best_model}_report.csv", index=False)
    print(f"Best report saved in 'output/4_task2/{best_model}_report.csv")

    print("\nRUN COMPLETED. Check the output folder for results and plots.")


if __name__ == "__main__":
    main()

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
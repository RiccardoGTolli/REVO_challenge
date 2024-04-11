# REVO - Data Challenge - Riccardo

## Import data

Import data into the mssql database using the following commands.

```bash
docker compose up --build -d
docker compose exec app bash
```

From inside container, run command below to import all data

```bash
python3 code/import_data.py all
```


## Train a model

From inside container, run command below to train NGB_1 using parameters from a config file:

```bash
python3 code/train_asset.py --config-file code/train_config_1.json --config-set ngb_asset
python3 code/train_sample.py --config-file code/train_config_1.json --config-set ngb_sample
```


This file will run train.py with the arguments from a config file (e.g.train_config_1.json).
- You can change the values train_config_1.json, config_set_1 to change the config file or the config set.
- You can also change the parameters in train_config_1.json to run train.py with different parameters.


## Format code

From inside container, run `format` and `code-checks` before submitting a PR

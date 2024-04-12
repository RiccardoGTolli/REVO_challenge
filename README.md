# REVO - Data Challenge - Riccardo

The files for the first and second task are all contained in revo/notebooks.
To get access you should run the container in development mode i.e. in the docker-compose.yml, you should have target: development (default).

The third task consists in the Docker app itself and all the files in revo/modules.
You can inspect and execute the files in development mode just fine but if you want the docker container to auto-execute the main.py you should run the container with target: production.

## Run the container
In the same directory where the docker-compose.yml is, open a linux terminal, on Windows you can use WSL(Windows Subsystem for Linux) and run:

```bash
docker compose up --build -d
```

## Execute Jupyter cells
Jupyter is exposed on port 8888, to access the service you need the token.

How to get the token:
```bash
docker compose exec revo jupyter server list
```
The token will change everytime the container is restarted.

method 1: Access Jupyter in browser
- Go to localhost:8888
- Type in the token

method 2: Within the IDE e.g. VsCode
- In a .ipynb file select the Kernel in the top right
- Select Existing Jupyter Server
- Type in the URL of the Jupyter Server which will be:
  http://localhost:8888/?token=<your_token>
- Type a name, the default 'localhost' is fine
- Select Python 3 (ipykernel)



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

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
  
## Gain access to container shell
Once the container is running (with docker compose up) you can gain access to the container shell with:

```bash
docker compose exec revo bash
```

## Run the app from start to finish

Method 1:
Build and run the image in production mode i.e., in the docker-compose.yml, have: 'target: production'
```bash
docker compose up --build -d
```

Method 2: Gain access to container shell and execute main.py manually.
With 'target: development': 
```bash
docker compose up --build -d
docker compose exec revo bash
```
Then execute main.py manually:
```bash
python3 modules/main.py --config modules/config.json
```
You can play with the arguments in config.json to obtain different model accuracy and change the start and end period. Simply change the values in config.json and rerun the app.
## Formatting, linters and static-checks (just for development purposes)

From inside container, run `format` and `code-checks`.

## Description of the app
# Petal

This repository contains the code for my Master Thesis.
Petal is a webservice for collecting and managing electrical signals from plants.
An instantiation of the system can be found at plant.biolingo.org/.
In the *src* directory the code for the web server can be found.
The *ml* directory contains the code for the developed machine learning models.

## Environment variables

Create .env file in root of directory
The following env variables can/need to be configured

- PROFILE: "prod" on server; "dev" locally
- JWT_SECRET_KEY: secret key for generation of password hashes
- POSTGRES_USER: user of postgres database
- POSTGRES_PASSWORD: password for postgres database
- POSTGRES_DB: name of postgres database
- DROPBOX_APP_KEY: app key of dropbox developer app
- DROPBOX_APP_SECRET: app secret of dropbox developer app
- DROPBOX_REFRESH_TOKEN: refresh token for dropbox developer app
- DELETE_MEASUREMENTS_AFTER_STOP: whether measurements should be deleted after recording is stopped
- DELETE_OBSERVATIONS_AFTER_STOP: whether observations should be deleted after experiment is finished
- MERGE_OBSERVATIONS_THRESHOLD: threshold for merging observations in milliseconds

## Running locally

Assumes environment variables have been configured

1. Create an environment: ``python -m venv venv``
2. Activate environment: ``source venv/bin/activate``
3. Install requirements: ``pip install -r requirements.txt``
4. Run: ``bash run-debug.sh``

## Deployment

1. Build docker image with ``docker build --network=host . <repository/image_name>``
2. Copy docker-compose.yml over to server
3. Adapt image field of plant-server service in docker-compose.yml to ``<repository/image_name>``
4. Add .env file on server
5. On server run: ``docker-compose up``

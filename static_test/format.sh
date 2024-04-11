#!/bin/bash

LINE_LENGTH=120

DIRS=(/src/python_code/modules /src/python_code/common_modules)

for DIR in "${DIRS[@]}"; do
    isort --profile black $DIR
    black -l $LINE_LENGTH $DIR
done



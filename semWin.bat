@echo off
REM Creates the python environment from requirements.yaml

set ENVIROMENT_YML = requirements.yaml

conda env create -f "%~dp0%ENVIROMENT_YML%"

conda activate CD_SEM

jupyter notebook
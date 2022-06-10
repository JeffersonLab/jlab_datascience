# General Data Science Toolkits 

## Software Requirement

- Python 3.9
- Additional python packages are defined in the setup.py
- This document assumes you are running at the top directory

## Directory Organization

```
├── environment.yml                   : Conda setup file with package requirements
├── setup.py                          : Python setup file with requirements files
├── core                	          : folder with dnc2s_rl code
    └── __init__.py                   : make base classes visible
    ├── keras_model         	      : folder containing models
        └── __init__.py               : make base classes visible
        └── siamese_model.py          : agent base class

```



## Installing

- Pull code from repo

```
git clone https://github.com/JeffersonLab/jlab_datascience.git
cd jlab_datascience
```
* Dependencies are managed using the conda environment setup:
```
conda env create -f environment.yml 
conda activate jlab_datascience (required every time you use the package)
```
* Install Data Science Toolkit (via pip):
```
pip install -e . 
```

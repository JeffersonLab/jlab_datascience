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
    ├── keras_models         	      : folder containing models
        └── __init__.py               : make base classes visible
        └── siamese_models.py         : agent base class

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

## If you plan to use examples, you need to install these additional dependency
* Install SUF SNS dependancy (via pip):
```
git clone https://code.ornl.gov/ml/lib/binary-storage.git
cd binary-storage/py/
pip install -e . 
```

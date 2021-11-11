# MLExp
resources for machine learning experiments

[![](https://github.com/OtsuKotsu/MLExp/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/OtsuKotsu/MLExp/actions/workflows/ci_cd.yml)

## Installation
### Install dependencies of this repository
#### use [Poetry](https://python-poetry.org/) (highly recommend to use this package manager)  
- only do `poetry install`  
- (to install [Poetry](https://python-poetry.org/), see [Poetry's official instruction](https://python-poetry.org/docs/master/). It's VERY EASY.)  
- if you want to install them without development dependencies,  
  do `poetry install --no-dev`
- see also [here](https://python-poetry.org/docs/cli/) to know about commands of Poetry.

#### use `pip`
- do `python3 -m pip install -r path/to/requirements.txt`
  - you cannot get development dependencies with this command

### Install this repository as the package `mlexp`
- `python3 -m pip install git+https://github.com/OtsuKotsu/MLExp`

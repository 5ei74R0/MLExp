# MLExp
resources for machine learning experiments

[![](https://github.com/OtsuKotsu/MLExp/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/OtsuKotsu/MLExp/actions/workflows/ci_cd.yml)

## Installation
### Install this repository as the package `mlexp`
- `python3 -m pip install git+https://github.com/OtsuKotsu/MLExp`

### Install dependencies of this repository
#### Use [Poetry](https://python-poetry.org/) (highly recommend to use this package manager)  
- Only do `poetry install`  
  (To install [Poetry](https://python-poetry.org/), see [Poetry's official instruction](https://python-poetry.org/docs/master/). It's VERY EASY.)  
- If you want to install them without development dependencies,  
  Do `poetry install --no-dev`
- See also [here](https://python-poetry.org/docs/cli/) to know about commands of Poetry.

#### Use `requirements.txt`
- Do `python3 -m pip install -r path/to/requirements.txt`
  - You cannot isolate development dependencies with this command

## Get started!
### Minimum usage
#### Decorator `mlexp.experiment.run` for Metrics Observation


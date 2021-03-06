# FeatureSelection_BuildingEnergy
Author: Liang Zhang@National Renewable Energy Laboratory

Hybrid feature selection method for building energy forecasting described [here](https://www.sciencedirect.com/science/article/pii/S0378778818321625)

Please cite the paper if you use the code for publication:

Zhang, L., & Wen, J. (2019). A systematic feature selection procedure for short-term data-driven building energy forecasting model development. Energy and Buildings, 183, 428-442.

This project is a work-in-progress.

Installation
Download and install the latest version of [Conda](https://docs.conda.io/en/latest/) (version 4.4 or above)
Create a new conda environment:

`$ conda create -n <name-of-repository> python=3.6 pip`

`$ conda activate <name-of-repository>`

(If you’re using a version of conda older than 4.4, you may need to instead use source activate <name-of-repository>.)

Make sure you are using the latest version of pip:

`$ pip install --upgrade pip`

Install the environment needed for this repository:

`$ pip install -e .[dev]`

After installing the environment, run the example python script "run/example.py". The example data is stored in "data/example_data.csv".

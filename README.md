# dat-sci-master-thesis
Materials of my Data Science Master's thesis


This repo contains submodule, so do git clone recursively:

`git clone --recursive https://github.com/kristjanr/dat-sci-master-thesis.git`


Install deps for submodule (OSX):

`conda env create -f donkeycar/install/envs/mac.yml` (this may take up to 5 minutes)

`conda activate donkey`

`pip install -e donkeycar/.`


Install deps for this project

`conda install -c conda-forge jupyterlab ipykernel scikit-learn nbformat=4.4.0`

`ipython kernel install --user --name=donkey`

`jupyter lab`

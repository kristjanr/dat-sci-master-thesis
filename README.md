![Driving Speed as a Hidden Factor Behind Distribution Shift - Tartu University 2022, Msc](visualabstract.png)

## Project setup
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

## Getting the predictions for open-loop evaluation

The [PredictAndSave](PredictAndSave.ipynb) notebook downloads data and models. Then it runs the inference, saving the results into a file.
The resulting files are already in this repository, in the [open-loop-results](open-loop-results) folder.

## Ground Truth Analysis - Chapters 3.4.1 and 4.1.3

...is in the [GroundTruthsAnalysis](GroundTruthsAnalysis.ipynb) notebook.
It creates the Figure 6. in Chapter 3.4.1, Automated Data Gathering and
Figure 11. in Chapter 4.1.3, Ground Truth Turning Angle Distribution.


## Validating Assumptions About Data - Chapter 4.1

The [ValidatingAssumptionsAboutDataI](ValidatingAssumptionsAboutDataI.ipynb) notebook provides results for 4.1.1 Driving Speed Differences and 4.1.2 Frame Differences chapters 



## Open-loop Evaluation - Chapter 4.2

The [OpenLoopEvaluation](OpenLoopEvaluation.ipynb) notebook calculates Mean Absolute Error for each of the model-speed - data-speed combinations. Results are used in Tables 1, 2, 3 in Chapter 4.2 in the thesis.
It also creates Figures 12. and 13. in Chapter 4.2.


## Chapter 4.4

[SaveActivations](ood/SaveActivations.ipynb) notebook downloads and loads the models and runs the inference, saving the activations used for OOD detection in Chapter 4.4
The resulting npy files are already saved in the [ood](ood) folder.

TODO for Chapter 4.4 - notebooks in the [ood](ood) folder:
* [ood/AnalyzeOOD.ipynb](ood/AnalyzeOOD.ipynb)
* [ood/clustering_dbscan.ipynb](ood/clustering_dbscan.ipynb)
* [ood/kmeans_tsne.ipynb](ood/kmeans_tsne.ipynb)
* [ood/Mahalanobis_and_AUROC.ipynb](ood/Mahalanobis_and_AUROC.ipynb)
![Driving Speed as a Hidden Factor Behind Distribution Shift - Tartu University 2022, Msc](visualabstract.png)

The thesis can be found [here](https://comserv.cs.ut.ee/ati_thesis/datasheet.php?id=75358&language=en).

## Project setup
This repo contains submodule, so do git clone recursively:

`git clone --recursive https://github.com/kristjanr/dat-sci-master-thesis.git`

Switch the submodule repository to dev branch:
`cd dat-sci-master-thesis/donkeycar`
`git switch dev`


Install deps for submodule (OSX):

`cd ..`

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

## Chapters 3.4.1 and 4.1.3 - Ground Truth Analysis
The  [GroundTruthsAnalysis](GroundTruthsAnalysis.ipynb) notebook creates 
- Figure 6. in Chapter 3.4.1, Automated Data Gathering 
- Figure 11. in Chapter 4.1.3, Ground Truth Turning Angle Distribution.


## Chapter 4.1 - Validating Assumptions About Data

The [ValidatingAssumptionsAboutDataI](ValidatingAssumptionsAboutDataI.ipynb) notebook provides results for 
- 4.1.1 Driving Speed Differences  
- 4.1.2 Frame Differences chapters 


## Chapter 4.2 - Open-loop Evaluation

The [OpenLoopEvaluation](OpenLoopEvaluation.ipynb) notebook calculates Mean Absolute Error for each of the model-speed - data-speed combinations. 
- Results are used in Tables 1, 2, 3 in Chapter 4.2 in the thesis.
- It also creates Figures 12. and 13. in Chapter 4.2.


## Chapter 4.4 - Multi-frame Inputs Become Out of Distribution

[SaveActivations](ood/SaveActivations.ipynb) notebook downloads and loads the models. Then it runs the inference, saving the activations used for OOD detection in Chapter 4.4.
The resulting npy files are already saved in the [ood](ood) folder.

### Chapter 4.4.1 - Activation Skewness 
The [ood/ActivationSkewness](ood/ActivationSkewness.ipynb) notebook creates
- Table 6. Basic statistics for activations
- Figure 15. Frame skewness AUROC


### Chapter 4.4.2 - Mahalanobis Distance and 4.4.3 T-distributed Stochastic Neighbour Embedding

[ood/KmeansMahalanobisAUROCtSNE](ood/KmeansMahalanobisAUROCtSNE.ipynb)
* Creates Figure 16. Finding the best number of clusters for K-means clustering method in Chapter 4.4.2.
* Creates Figure 17. ROC curve for InD and OOD Mahalanobis distances from K-means cluster centres in Chapter 4.4.2.
* Creates Figure 18. Fast model's t-SNE 3D plot viewed from three different angles in Chapter 4.4.3.

## Chapter 4.5 - Multi-frame Input Distribution Shift Analysis Using Synthesised Data 
The [synthesized](synthesized) folder contains the notebooks which repeat the previous findings for multi-frame models.
For example, the "Figure 19. Fast synthesized data model's activations t-SNE 3D plot viewed from three different angles." is created in the [synthesized/KmeansMahalanobisAUROCtSNE.ipynb](synthesized/KmeansMahalanobisAUROCtSNE.ipynb) notebook.  

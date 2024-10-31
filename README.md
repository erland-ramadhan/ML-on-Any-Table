<div align="center">

# ML on Any Table

<div align="left">

<!-- <h2 id="first-section">What is this repo about?</h2> -->
## What is this repo about?

This repository contains a web application for building and deploying a Random Forest model, with options for both regression and classification tasks. The app uses Python's Flask for the backend and HTML, CSS, and JavaScript for the frontend.

Users can upload a "cleaned" dataset in CSV or XLSX format. This dataset should contain no missing values or prior transformations. Once uploaded, the app generates a Cramér’s V heatmap to visualize correlations among categorical features.

Based on the heatmap, users can specify how to encode the categorical features. With encoding selected, the app trains a Random Forest model, splits the data, and displays performance metrics.

The trained model is saved and can make predictions on a single-row CSV file, excluding the target column.

## Before using this repo...

First, clone this repo to your local machine. Then, create a new conda environment with its dependencies by executing the following syntax in terminal or console.

> Note: This repo is accelerated with Intel(R) Extension for Scikit-learn. So, make sure to clone this repo on Intel-powered local machine.

```
conda env create --file requirements.yml
```

After a new conda environment created, activate the environment by

```
conda activate sars-cov-2_clustering
```

Fuzzy C-Means clustering method is still not available in conda package manager. While in the activated `sars-cov-2_clustering` environment, install fuzzy-c-means package with `pip`.

```
pip install fuzzy-c-means
```

## How to use this repo?

First, as stated in [What is this repo about?](#first-section), a $k$-mers-based approach is performed by executing the following syntax on terminal/console.

> Ensure that your current working directory is set to the cloned repo.

```
python featureVec_Generator.py
```

After generating and saving feature vectors in a NumPy binary files, clustering can be performed using the methods in each respective folder.

Each respective clustering method folder includes the following:
- org/
  - Clustering without feature selection applied to the feature vectors.
  - Results in a longer runtime and lower $F_1$ score due to high dimensionality.
- Boruta/
  - Clustering after applying Boruta.
  - Boruta is a supervised method that is made around the random forest (RF) classification algorithm.
  - This works by creating shadow features so that the features do not compete among themselves, but rather they compete with randomized version of them.
    - It then extracts the importance of each feature (corresponding to the class label) and only keeps the features that are above specific threshold of importance.
- Lasso/
  - Clustering after applying **L**east **A**bsolute **S**hrinkage and **S**election **O**perator (LASSO) regression.
  - LASSO is a specific case of the penalized least squares regression with $L_1$ penalty function.
  - By combining the good qualities of ridge regression and subset selection, LASSO can improve both model interpretability, and prediction regression.
- RFT/
  - Clustering after applying a kernel-based method called Random Fourier Features (RFT).
  - RFT is an unsupervised approach, which maps the input data to a randomized low dimensional feature space (Euclidean inner product space) to get an approximate representation of data in lower dimensions $D$  from the original dimensions $d$.

To perform clustering on feature vectors, run:

```
python clusteringMethod_featureSelection.py
```

Here, replace `clusteringMethod` with `kmeans`, `kmodes`, or `fuzzy` and `featureSelection` with `org`, `Boruta`, `Lasso`, or `RFT`.

Next, create a contingency table by executing:

```
python new_cnt.py
```

Finally, calculate $F_1$ scores of the clustering method by running:

```
python Calculate_F1.py
```

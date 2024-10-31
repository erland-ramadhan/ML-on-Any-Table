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

First, clone this repo to your local machine. Then, create a new Conda environment with its dependencies by executing the following command in the terminal or console.

> Note: This repo is accelerated with Intel(R) Extension for Scikit-learn. So, make sure to clone this repo on an Intel-powered local machine.

```
conda env create --file requirements.yml
```

After the environment is created, activate it with:

```
conda activate ml-any-table
```

## How to use this repo?

To launch the web application, execute the following command in the terminal/console.

> Ensure that your current working directory is set to the cloned repo.

```
python backend.py
```

Once the `Debugger is activated!` message appear in the terminal, open the browser (Chrome, Firefox, Safari, etc.) and go to `localhost:8000`. The web application is now ready to use!

### Demo

The tutorial video on how to use the web app is provided below.

https://github.com/user-attachments/assets/a6774cb7-c922-4785-84dc-5b524dd276d1

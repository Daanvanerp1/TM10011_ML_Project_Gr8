# TM10011 Machine Learning Project: GIST Radiomics Classification - Group 8

## Project Overview
This repository contains the code for the final research project of the TM10011 Machine Learning course. The objective is to develop and evaluate a machine learning pipeline to classify gastrointestinal stromal tumors (GIST) based on radiomic features extracted from medical images.

## Dataset
The project utilizes the `GIST_radiomicFeatures.csv` dataset, which contains various radiomic features and a binary target label (GIST and Non-GIST).

## Methodology
To ensure an unbiased evaluation of the model performance, a Nested Cross-Validation approach is implemented. The pipeline systematically evaluates multiple feature selection methods and classification algorithms.

The full preprocessing and classification pipeline consists of:
* **Imputation:** Median imputation for missing values.
* **Variance Thresholding:** Removal of zero-variance features.
* **Scaling:** RobustScaler to handle existing outliers gracefully.
* **Outlier Handling:** Custom Winsorizer to cap extreme outliers.
* **Feature Selection:** Evaluation of multiple methods including a custom Correlation Filter, Mann-Whitney U-Test, LASSO, and Recursive Feature Elimination (RFE).
* **Classifiers:** Logistic Regression, Random Forest, Support Vector Machine (SVM), and XGBoost.

Hyperparameter tuning is performed using `RandomizedSearchCV` within the inner loops of the cross-validation.

## Installation and Setup
To run this project locally, ensure you have Python installed. 

1. Clone this repository:
   git clone https://github.com/Daanvanerp1/TM10011_ML_Project_Gr8.git

2. Navigate to the project directory:
   cd TM10011_ML_Project_Gr8.git

3. Install the required dependencies:
   pip install -r requirements.txt

## Usage
To execute the machine learning pipeline, evaluate the folds, and train the final model, run the main script:

python Final_assignment_file.py

The script will output the performance metrics for each outer fold, display the cumulative confusion matrix, and print the hyperparameters of the final model trained on the complete dataset.

## Authors
Daan van Erp (5505267)
Justus Theirry (5592720)
Anna Dijk (5807018)
Julie Rongen (5276594)
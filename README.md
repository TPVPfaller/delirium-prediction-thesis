# Prediction of ICU Delirium using Machine Learning Models

## Data
The dataset consists of 18,270 ICU admissions from the eICU Collaborative Research Database (https://www.nature.com/articles/sdata2018178).

If this dataset isn't accessible, there are two additional datasets that can be used:
 - titanic.csv: Small dataset for fast execution times
 - heart_disease.csv: Large dataset

## Setup

1. Create conda enveironment and install packages:
```python
pip install -r requirements.txt
```
2. For faster training times of the FT-Transformer install Cuda:
```python
conda install -c conda-forge cudatoolkit
```
```python
conda install -c conda-forge cudnn
```


## EDA.py
Exploratory data analysis is used to summarize the dataset's characteristics by generating plots and tables.

Implemented methods:
- Spearman and Pearson correlation returning the top correlations of the dataset.
- Cramers_V for calculating the correlation of two categorical variables.
- plot_corr(pearson_df, spearman_df): bar plot for plotting Pearson and Spearman correlation next to each other.
- pie(df): pie chart of predictor distribution
- show_corr(df, var1, var2): visualizes correlation of two variables.
  - two categorical variables: two pie charts
  - one categorical, one numerical: two violin plots
  - two numerical: scatter plot

## models.py
Implementations for Logistic Regression, CatBoost, and FT-Transformer.

- hpo_cv(X, y, model, param_space, folds=5, n_trials=50): Performs Optuna hyperparameter optimization 
with the specified hyperparameter space using stratified cross-validation. Returns the best parameters after n_trials.
- train_and_evaluate(train_and_evaluate(X_train, y_train, X_test, y_test, clf, parameters, name):
  - Uses bootstrapping on the test prediction for creating confidence intervals
  - Outputs the following plots after training and testing:
    - ROC curves
    - Precision-Recall curves
    - Shapley summary plot or importances inferred from attention weights in the case of the FT-Transformer
    - Confusion matrix
    - Calibration curve
- borutaShap_feature_selection(X_test, y_test, clf): takes a CatBoost classifier and outputs the all-relevant features. Outputs box plots for the all-relevant features and the shadow attributes.
- boruta_feature_selection(X, y, max_iter=100): does normal boruta feature selection using a Random Forest Classifier (less accurate).
- catboost_fs(clf, X_train, X_test, y_train, y_test): Recursive feature selection for CatBoost using Shapley values. Selects a specified number of features and outputs a graph showing the loss for every removed feature.


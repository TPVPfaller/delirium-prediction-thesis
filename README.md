# Prediction of ICU Delirium using Machine Learning Models

 The dataset consists of 18,270 ICU admissions from the eICU Collaborative Research database 
 (https://www.nature.com/articles/sdata2018178).

## EDA.py
Exploratory data analysis for summarizing characteristics of the dataset by generating plots and tables.

Implemented methods:
- Spearman and Pearson correlation returning the top correlations of the dataset.
- Cramers_V for calculating the correlation of two categorical variables.
- plot_corr(pearson_df, spearman_df): bar plot for plotting Pearson and Spearman correlation next to each other.
- PCA (pca_vis(df)) and t-SNE (tsne_vis(df))
- pie(df): pie chart of predictor distribution
- show_corr(df, var1, var2): visualizes correlation of two variables.
  - two categorical variables: two pie charts
  - one categorical, one numerical: two violin plots
  - two numerical: scatter plot

## models.py
Implementations for Logistic regression, CatBoost, and FT-Transformer.

- hpo_cv(X, y, model, param_space, folds=5, n_trials=50): Performs Optuna hyperparamter optimization 
with the specified hyperparameter space using stratified cross validation. Returns the best parameters after n_trials.
- train_and_evaluate(train_and_evaluate(X_train, y_train, X_test, y_test, clf, parameters, name):
  - 
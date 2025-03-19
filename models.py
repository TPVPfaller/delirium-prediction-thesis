# model train and validate & evaluation
import tensorflow as tf
from sklearn.calibration import calibration_curve
import json

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import numpy as np
from sklearn.utils import shuffle
from boruta import BorutaPy
import shap
import statistics
import shap.maskers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, \
    auc, fbeta_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import t
import pandas as pd
import optuna
from BorutaShap import BorutaShap
import matplotlib
from catboost import CatBoostClassifier, EFeaturesSelectionAlgorithm, EShapCalcType
import itertools
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from tabtransformertf.utils.preprocessing import df_to_dataset
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer


def cv_hpo(X, y, clf, param_space, folds=5, n_trials=150):
    study = optuna.create_study(
        study_name='study',
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )

    def suggest_params(trial):
        return {
            name: (
                trial.suggest_categorical(name, values) if isinstance(values[0], str) else
                trial.suggest_int(name, values[0], values[1]) if isinstance(values[0], int) else
                trial.suggest_float(name, values[0], values[1], log=True) if "log" in values else
                trial.suggest_float(name, values[0], values[1])
            )
            for name, values in param_space.items()
        }

    def objective(trial):
        clf.set_params(**suggest_params(trial))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
        return cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', error_score="raise").mean()

    def objective_ft(trial):
        params = suggest_params(trial)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
        roc_auc_scores = []

        for train_i, test_i in cv.split(X, y):
            X_train, X_test = X.iloc[train_i], X.iloc[test_i]
            y_train, y_test = y.iloc[train_i], y.iloc[test_i]
            ft_clf, early_stopping, weighted = ft_transformer(X_train, params, clf[2])
            class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

            train_dataset, val_dataset = create_train_val_datasets(X_train, y_train)
            ft_clf.fit(
                train_dataset,
                epochs=50,
                validation_data=val_dataset,
                callbacks=[early_stopping],
                class_weight={0: class_weights[0], 1: class_weights[1]} if weighted else None,
            )
            auroc = roc_auc_score(y_test, ft_clf.predict(format_df(X_test, y_test))['output'])
            print("AUROC: " + auroc)
            roc_auc_scores.append(auroc)

        return statistics.mean(roc_auc_scores)

    study.optimize(objective_ft if isinstance(clf, tuple) else objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    return best_params


def best_f2_threshold(y_probs, y_true):
    thresholds = np.linspace(0, 1, 100)
    f2_scores = [fbeta_score(y_true, (y_probs >= t).astype(int), beta=2) for t in thresholds]

    best_idx = np.argmax(f2_scores)
    return f2_scores[best_idx], thresholds[best_idx]


def train_and_evaluate(X_train, y_train, X_test, y_test, clf, parameters, name):
    precision_scores, recall_scores, roc_auc_scores, pr_auc_scores = [], [], [], []
    brier_scores, f2_scores, specificity_list, sensitivity_list = [], [], [], []
    npv_list, precision_list, recall_list, fpr_list, tpr_list = [], [], [], [], []

    if type(clf) == tuple:
        clf, early_stopping, weighted = clf
        weights = compute_class_weight('balanced', np.unique(y_train), y_train)
        train_dataset, val_dataset = create_train_val_datasets(X_train, y_train)
        clf.fit(
            train_dataset,
            epochs=50,
            validation_data=val_dataset,
            callbacks=[early_stopping],
            class_weight={0: weights[0], 1: weights[1]} if weighted else None,
        )
        res = clf.predict(format_df(X_test, y_test))
        y_pred_proba = res['output']
        try:
            plot_ft_importances(res['importances'], X_train)
        except Exception as e:
            print("Error plotting importances:", e)
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        clf.set_params(**parameters)
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

    if type(y_pred[0]) == str:
        y_pred = y_pred == 'True'

    rng, idx = np.random.RandomState(seed=2), np.arange(y_test.shape[0])
    y_test = y_test.reset_index(drop=True).to_numpy().astype(int)

    for i in range(200):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        precision, recall, _ = precision_recall_curve(y_test[pred_idx], y_pred_proba[pred_idx])
        pr_auc_scores.append(auc(recall, precision))
        precision_scores.append(precision)
        recall_scores.append(recall)
        roc_auc_scores.append(roc_auc_score(y_test[pred_idx], y_pred_proba[pred_idx]))
        brier_scores.append(brier_score_loss(y_test[pred_idx], y_pred_proba[pred_idx]))
        f2_scores.append(fbeta_score(y_true=y_test[pred_idx], y_pred=y_pred[pred_idx], beta=2))

        tn, fp, fn, tp = confusion_matrix(y_test[pred_idx], y_pred[pred_idx]).ravel()
        precision_list.append(precision_score(y_test[pred_idx], y_pred[pred_idx]))
        recall_list.append(recall_score(y_test[pred_idx], y_pred[pred_idx]))
        specificity_list.append(tn / (tn + fp))
        sensitivity_list.append(tp / (tp + fn))
        npv_list.append(tn / (tn + fn))

        fpr, tpr, _ = roc_curve(y_test[pred_idx], y_pred_proba[pred_idx])
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    print_metrics_with_ci(roc_auc_scores, pr_auc_scores, brier_scores, f2_scores, specificity_list, sensitivity_list,
                          npv_list, precision_list, recall_list)

    chance = len(y_test[y_test == 1]) / len(y_test)

    title = 'Non-weighted' if 'notbal' in name else 'Weighted'
    plot_curves(precision_scores, recall_scores, fpr_list, tpr_list, roc_auc_scores, pr_auc_scores, chance, title, name)
    plot_curves(precision_scores, recall_scores, fpr_list, tpr_list, roc_auc_scores, pr_auc_scores, chance, title, name)
    plot_calibration(y_pred_proba, y_test, name)
    best_f2, f2_threshold = best_f2_threshold(y_pred_proba, y_test)
    print("Best F2 score: {:.3f} (at threshold {:.3f})".format(best_f2, f2_threshold))
    plot_confusion_matrix(clf, X_test, y_test, name, normalize=False, title=title)
    if 'ft' not in name:
        get_shap(clf, X_test, name)
    return clf


def print_metrics_with_ci(roc_auc, pr_auc, brier, f1, specificity_list, sensitivity_list, npv_list,
                          precision_list, recall_list):
    mean_roc_auc, roc_auc_ci = calculate_confidence_interval(roc_auc)
    mean_pr_auc, pr_auc_ci = calculate_confidence_interval(pr_auc)
    mean_brier, brier_ci = calculate_confidence_interval(brier)
    mean_f2, f2_ci = calculate_confidence_interval(f1)
    specificity_mean, specificity_ci = calculate_confidence_interval(specificity_list)
    sensitivity_mean, sensitivity_ci = calculate_confidence_interval(sensitivity_list)
    npv_mean, npv_ci = calculate_confidence_interval(npv_list)
    precision_mean, precision_ci = calculate_confidence_interval(precision_list)
    recall_mean, recall_ci = calculate_confidence_interval(recall_list)

    print("ROC AUC Score: {:.3f} ({:.3f}, {:.3f})".format(mean_roc_auc, mean_roc_auc - roc_auc_ci,
                                                          mean_roc_auc + roc_auc_ci))
    print("PR AUC Score: {:.3f} ({:.3f}, {:.3f})".format(mean_pr_auc, mean_pr_auc - pr_auc_ci, mean_pr_auc + pr_auc_ci))
    print("Brier Score: {:.3f} ({:.3f}, {:.3f})".format(mean_brier, mean_brier - brier_ci, mean_brier + brier_ci))
    print("F2 Score: {:.3f} ({:.3f}, {:.3f})".format(mean_f2, mean_f2 - f2_ci, mean_f2 + f2_ci))
    print("Specificity: {:.3f} ({:.3f}, {:.3f})".format(specificity_mean, specificity_mean - specificity_ci,
                                                        specificity_mean + specificity_ci))
    print("Sensitivity: {:.3f} ({:.3f}, {:.3f})".format(sensitivity_mean, sensitivity_mean - sensitivity_ci,
                                                        sensitivity_mean + sensitivity_ci))
    print("Negative Pred Value (NPV): {:.3f} ({:.3f}, {:.3f})".format(npv_mean, npv_mean - npv_ci, npv_mean + npv_ci))
    print("Precision (PPV): {:.3f} ({:.3f}, {:.3f})".format(precision_mean, precision_mean - precision_ci,
                                                            precision_mean + precision_ci))
    print("Recall: {:.3f} ({:.3f}, {:.3f})".format(recall_mean, recall_mean - recall_ci, recall_mean + recall_ci))


def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std = statistics.stdev(data)
    t_value = t.ppf((1 + confidence) / 2., n - 1)
    margin_of_error = t_value * std
    return mean, margin_of_error


def plot_curves(precision_scores, recall_scores, fpr_list, tpr_list, roc_auc, pr_auc, chance, title, name):
    plt.style.use('seaborn-paper')

    mean_roc_auc, roc_auc_ci = calculate_confidence_interval(roc_auc)
    mean_pr_auc, pr_auc_ci = calculate_confidence_interval(pr_auc)

    recall_common = np.linspace(0, 1, 100)
    precision_interp = np.array(
        [np.interp(recall_common, r[::-1], p[::-1]) for p, r in zip(precision_scores, recall_scores)])

    lower_precision = np.percentile(precision_interp, 2.5, axis=0)
    upper_precision = np.percentile(precision_interp, 97.5, axis=0)

    mean_precision = np.mean(precision_interp, axis=0)
    plt.figure()
    plt.plot(recall_common, mean_precision, color='blue', lw=2,
             label=f'Mean PR ({mean_pr_auc:.3f} [{mean_pr_auc - pr_auc_ci:.3f}, {mean_pr_auc + pr_auc_ci:.3f}])')
    plt.fill_between(recall_common, lower_precision, upper_precision, color="orange", alpha=0.3, label="95% CI")
    plt.plot([0, 1], [chance, chance], color='red', lw=2, linestyle='--', label='Chance')
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title or 'Precision-Recall Curve', fontsize=19)
    plt.legend(loc="best", fontsize=13)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(f"plots/pr_{name}.pdf", format="pdf", bbox_inches="tight")

    fpr_common = np.linspace(0, 1, 100)
    tpr_interp = np.array([np.interp(fpr_common, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)])

    lower_tpr = np.percentile(tpr_interp, 2.5, axis=0)
    upper_tpr = np.percentile(tpr_interp, 97.5, axis=0)

    mean_tpr = np.mean(tpr_interp, axis=0)
    plt.figure()
    plt.plot(fpr_common, mean_tpr, color='blue', lw=2,
             label=f'Mean ROC ({mean_roc_auc:.3f} [{mean_roc_auc - roc_auc_ci:.3f}, {mean_roc_auc + roc_auc_ci:.3f}])')
    plt.fill_between(fpr_common, lower_tpr, upper_tpr, color="orange", alpha=0.3, label="95% CI")
    plt.plot([0, 1], [0, 1], color='red', lw=2, label="Chance", linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title or 'Receiver Operating Characteristic (ROC) Curve', fontsize=19)
    plt.legend(loc="lower right", fontsize=13)
    plt.savefig(f"plots/roc_{name}.pdf", format="pdf", bbox_inches="tight")


def get_importances(model, X_train):
    plt.style.use('seaborn-paper')
    ftr_importances = model.feature_importances
    ftr_importances = pd.Series(ftr_importances, index=X_train.columns)
    ftr_imp = ftr_importances.sort_values(ascending=False)[:70]

    plt.figure(figsize=(10, 20))
    plt.title('Feature Importances')
    sns.barplot(x=ftr_imp, y=ftr_imp.index)
    plt.savefig('plots/feature_importances.pdf', format="pdf", bbox_inches="tight")
    plt.show()


def get_confusion_matrix(clf, X_test, y_test):
    if isinstance(clf, FTTransformer):
        y_pred = clf.predict(format_df(X_test, y_test))
        y_pred = (y_pred['output'] >= 0.5).astype(int)
    else:
        y_pred = clf.predict(X_test)
    if type(y_pred[0]) == str:
        y_pred = y_pred == 'True'
    return confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(clf, X_test, y_test, name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.style.use('seaborn-paper')
    classes = ['No delir', 'Delir']
    cm = get_confusion_matrix(clf, X_test, y_test)
    tp_fn_sum = np.sum(cm[:, -1])
    title_fontsize_first_line = 17

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + '\n' + f"TP+FN: {tp_fn_sum}", fontsize=title_fontsize_first_line)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=13)
    plt.yticks(tick_marks, classes, fontsize=13)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 3) * 100

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

    plt.savefig(f'plots/cm_{name}.pdf', format="pdf", bbox_inches="tight")


def plot_calibration(probs, y_test, name):
    plt.style.use('seaborn-paper')
    fop, mpv = calibration_curve(y_test, probs, n_bins=10)
    plt.figure()
    plt.title('Calibration Curve', fontsize=19)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(mpv, fop, marker='.')
    plt.savefig(f'plots/calibration_{name}.pdf', format="pdf", bbox_inches="tight")


def binarize_features(df, columns):
    for col in columns:
        df[col] = df[col].astype(bool)
    return df


def catboost_fs(clf, X_train, X_test, y_train, y_test, num_features=37):
    summary = clf.select_features(
        X_train,
        y=y_train,
        eval_set=(X_test, y_test),
        features_for_select='0-128',
        num_features_to_select=num_features,
        steps=10,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
        logging_level='Verbose',
        plot=True
    )
    print(summary)
    plt.figure()
    plt.style.use('seaborn-paper')
    plt.title('Recursive Feature Selection', fontsize=19)
    plt.plot(summary['loss_graph']['removed_features_count'], summary['loss_graph']['loss_values'], marker='o',
             markevery=summary['loss_graph']['removed_features_count'], markersize=4, color='green')
    plt.xlabel('Number of removed features', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('plots/cb_bal_fs.pdf', format="pdf", bbox_inches="tight")


# 8 Selected features: ['apachescore', 'age', 'rass_mean', 'temperature_mean', 'gcs_verbal_mean', 'gcs_total_mean', 'respiratoryrate_mean', 'heartrate_mean']
def boruta_feature_selection(X, y, max_iter=100):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced', random_state=1)

    boruta = BorutaPy(rf, verbose=2, random_state=1, max_iter=max_iter, alpha=0.3)
    boruta.fit(X.values, y)

    return X.columns[boruta.support_].tolist()


# Boruta all-relevant feature selection
def borutaShap_feature_selection(X, y, model):
    feat_selector = BorutaShap(
        model=model,
        importance_measure='shap',
        classification=True,
    )

    feat_selector.fit(X, y, n_trials=120, random_state=1, verbose=True)
    feat_selector.TentativeRoughFix()
    feat_selector.results_to_csv(filename='borutaShap_importance')
    feat_selector.plot(X_size=12, figsize=(12, 8),
                       y_scale='log', which_features='accepted', display=False)
    plt.savefig('plots/borutaShap.pdf', format="pdf", bbox_inches="tight")


def prepare_data():
    df = pd.read_csv("data/first24hourdata_new.csv")

    df.columns = [x.lower() for x in df.columns]
    df = df.drop(
        ['patientunitstayid', 'tstart', 'tend', 'interval', 'unnamed_0', 'dummy_1', 'dummy_2', 'dummy_3', 'delir',
         'offset', 'interval', 'cause'], axis=1, errors='ignore')
    # Identify binary columns
    binary_cols = df.columns[(df.nunique() == 2) & df.isin([0, 1]).all()]
    df[binary_cols] = df[binary_cols].astype(bool)
    return df


def split_df(df):
    df = shuffle(df, random_state=1)
    X = df[df['event'] != 'none']
    y = df['event']
    y = (y == 1)
    X = X.drop(['event'], axis=1)
    return X, y


def create_test_split(X, y, test_size=0.2):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=True, random_state=1
    )
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    # Apply feature scaling
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    X_train_scaled = pd.DataFrame(X_train, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test, columns=X_test.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test


def get_shap(clf, X, name):
    X = X * 1.0  # cast to float
    plt.figure()
    plt.style.use('seaborn-paper')
    plt.rcParams['font.size'] = 16
    if 'notbal' in name:
        plt.title('Non-weighted', fontsize=19)
    else:
        plt.title('Weighted', fontsize=19)
    if 'LR' in name:
        explainer = shap.LinearExplainer(model=clf, masker=shap.maskers.Independent(X))
    elif 'cb' in name:
        explainer = shap.TreeExplainer(model=clf, masker=shap.maskers.Independent(X))
    else:
        explainer = shap.KernelExplainer(clf, X.iloc[:1500, :])
        shap_values = explainer.shap_values(X.iloc[2000:5000, :])
        shap.summary_plot(shap_values, X, max_display=20, show=False)
        plt.savefig("plots/{}_shapley_plot.pdf".format(name), format="pdf", bbox_inches="tight")
        return
    shap_values = explainer.shap_values(X)
    shap_importances = get_shap_importance(X, shap_values)
    print(shap_importances.head(20))

    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.savefig("plots/{}_shapley_plot.pdf".format(name), format="pdf", bbox_inches="tight")


def format_df(X, y):
    df = X.copy()
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    df[cat_features] = df[cat_features].astype(str)  # Categorical as string
    df[num_features] = df[num_features].astype(np.float32)  # Numerical as float32

    df['event'] = y.astype(int)

    data = df_to_dataset(df, shuffle=False, batch_size=1024)

    return data

def create_train_val_datasets(X, y):
    df = X.copy()
    y = y.astype('int64')
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    df[cat_features] = df[cat_features].astype(str)
    df[num_features] = df[num_features].astype(float)

    train_data, val_data, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=1, stratify=y)

    train_data['event'] = y_train
    val_data['event'] = y_val

    train_dataset = df_to_dataset(train_data, 'event', shuffle=True, batch_size=64)
    val_dataset = df_to_dataset(val_data, 'event', shuffle=False, batch_size=64)
    return train_dataset, val_dataset
# [I 2025-03-16 21:07:57,359] Trial 13 finished with value: 0.7602067459840167 and parameters: {'embedding_dim': 129, 'encoder_depth': 8, 'heads': 12, 'attn_dropout': 0.10643661740462781, 'ff_dropout': 0.27226748919100047, 'transformer_depth': 10, 'learning_rate': 0.00022053183640516512, 'weight_decay': 6.109290242505328e-06}. Best is trial 13 with value: 0.7602067459840167.

def ft_transformer(X, params, weighted=True):
    df = X.copy()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    df[cat_features] = df[cat_features].astype(str)
    df[num_features] = df[num_features].astype(float)
    if params:
        encoder_params = {
            'numerical_features': num_features,
            'categorical_features': cat_features,
            'numerical_data': df[num_features].values,
            'categorical_data': df[cat_features].values,
            'embedding_dim': params['embedding_dim'],
            'depth': params['encoder_depth'],
            'heads': params['heads'],
            'attn_dropout': params['attn_dropout'],
            'ff_dropout': params['ff_dropout'],
            'explainable': True
        }
        lr, weight_decay = params['learning_rate'], params['weight_decay']
    else: # default Hyperparameters
        encoder_params = {
            'numerical_features': num_features,
            'categorical_features': cat_features,
            'numerical_data': df[num_features].values,
            'categorical_data': df[cat_features].values,
            'embedding_dim': 16,
            'depth': 6,
            'heads': 8,
            'attn_dropout': 0.2,
            'ff_dropout': 0.2,
            'explainable': True
        }
        lr, weight_decay = 0.0001, 0.00001

    ft_linear_encoder = FTTransformerEncoder(**encoder_params)
    ft_model = FTTransformer(encoder=ft_linear_encoder, out_dim=1, depth=params.get('transformer_depth', 6),
                             out_activation='sigmoid')
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)

    ft_model.compile(
        optimizer=optimizer,
        loss={"output": tf.keras.losses.BinaryCrossentropy(), "importances": None},
        metrics={"output": [tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]},
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

    return ft_model, early_stopping, weighted


def plot_ft_importances(importances, X):
    feature_names = X.columns
    importances = importances[:, :-1]

    mean_importances = np.mean(importances, axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_importances
    })

    top_20 = importance_df.sort_values(by="Importance", ascending=False).head(20)

    plt.style.use("seaborn-paper")
    plt.figure(figsize=(8, 10))

    plt.barh(top_20["Feature"], top_20["Importance"], color='b', alpha=0.7)

    plt.xlabel("Mean Importance", fontsize=18)
    plt.ylabel("Feature Name", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Top 20 Feature Importances", fontsize=24)
    plt.gca().invert_yaxis()  # Invert y-axis to show highest at top

    plt.tight_layout()
    plt.savefig("plots/ft_importances.pdf", format="pdf", bbox_inches="tight")


def get_shap_importance(X_val, shap_values):
    df_shap_values = pd.DataFrame(data=shap_values, columns=X_val.columns)
    df_feature_importance = pd.DataFrame(columns=['feature', 'importance'])

    for col in df_shap_values.columns:
        importance = df_shap_values[col].abs().mean()
        df_feature_importance.loc[len(df_feature_importance)] = [col, importance]

    return df_feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    plt.style.use('seaborn-paper')
    matplotlib.rcParams.update({'figure.autolayout': True})

    # Load parameters and feature set from JSON files
    params = load_json('params.json')
    features = load_json('features.json')

    smaller_featureset = features['feature_set']
    df = prepare_data()

    df = df[smaller_featureset]
    X, y = split_df(df)

    LR_space = params['LR_space']
    LR_bal_params = params['LR_bal_params']
    LR_notbal_params = params['LR_notbal_params']
    fs_LR_bal_params = params['fs_LR_bal_params']
    fs_LR_notbal_params = params['fs_LR_notbal_params']

    cb_space = params['cb_space']
    cb_bal_params = params['cb_bal_params']
    cb_notbal_params = params['cb_notbal_params']
    fs_cb_bal_params = params['fs_cb_bal_params']
    fs_cb_notbal_params = params['fs_cb_notbal_params']

    ft_space = params['ft_space']
    ft_bal_params = params['ft_bal_params']
    fs_ft_bal_params = params['fs_ft_bal_params']

    LR_notbal_clf = LogisticRegression(n_jobs=-1, random_state=1)
    LR_bal_clf = LogisticRegression(class_weight='balanced', n_jobs=-1, random_state=1)

    cb_bal_clf = CatBoostClassifier(auto_class_weights='Balanced', verbose=False, eval_metric='AUC', random_state=1)
    cb_notbal_clf = CatBoostClassifier(verbose=False, eval_metric='AUC', random_state=1)

    ft_bal_clf = ft_transformer(X, fs_ft_bal_params, weighted=True)
    ft_notbal_clf = ft_transformer(X, fs_ft_bal_params, weighted=False)

    X_train, X_test, y_train, y_test = create_test_split(X, y, 0.2)

    #best_params = cv_hpo(X_train, y_train, LR_bal_clf, LR_space, folds=3)

    clf = train_and_evaluate(X_train, y_train, X_test, y_test, cb_bal_clf, fs_cb_bal_params, name="cb_bal")

    #borutaShap_feature_selection(X_test, y_test, clf)

    # cb_bal_clf.set_params(**best_cb_bal_params)
    # catboost_fs(cb_bal_clf, X_train, X_test, y_train, y_test)

    plt.show()


if __name__ == '__main__':
    main()

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


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def prepare_data(location="data/first24hourdata_new.csv", target='event', drop_right_censored=False):
    df = pd.read_csv(location).dropna().rename(columns={target: 'target'})
    if drop_right_censored:
        df = df[df['target'] != 0]
    df['target'] = df['target'] == 1
    df.columns = df.columns.str.lower()

    drop_cols = ['patientunitstayid', 'tstart', 'tend', 'interval', 'unnamed_0',
                 'dummy_1', 'dummy_2', 'dummy_3', 'delir', 'offset', 'cause']
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    for col in df.select_dtypes(include=['object', 'string']):
        df[col] = df[col].astype('category').cat.codes

    binary_cols = df.columns[(df.nunique() == 2) & df.isin([0, 1]).all()]
    df[binary_cols] = df[binary_cols].astype(bool)
    return df


def split_df(df):
    df = shuffle(df, random_state=1)
    return df.drop('target', axis=1), df['target']


def create_test_split(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y, shuffle=True, random_state=1)
    num_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return pd.DataFrame(X_train), pd.DataFrame(X_test), y_train, y_test


def cast_num_cat_features(df):
    df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    df[cat_cols] = df[cat_cols].astype(str)
    return df, num_cols, cat_cols


def create_train_val_datasets(X, y):
    df, num_cols, cat_cols = cast_num_cat_features(X.copy())
    y = y.astype('int64')
    train_data, val_data, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=1, stratify=y)
    train_data['target'], val_data['target'] = y_train, y_val

    return (
        df_to_dataset(train_data, 'target', shuffle=True, batch_size=64),
        df_to_dataset(val_data, 'target', shuffle=False, batch_size=64)
    )


def format_df(X, y):
    df, _, _ = cast_num_cat_features(X.copy())
    df['target'] = y.astype(int)
    return df_to_dataset(df, shuffle=False, batch_size=1024)


def ft_transformer(X, params=None, weighted=True):
    df, num_cols, cat_cols = cast_num_cat_features(X.copy())
    encoder_params = {
        'numerical_features': num_cols,
        'categorical_features': cat_cols,
        'numerical_data': df[num_cols].values,
        'categorical_data': df[cat_cols].values,
        'embedding_dim': params.get('embedding_dim', 16),
        'depth': params.get('encoder_depth', 6),
        'heads': params.get('heads', 8),
        'attn_dropout': params.get('attn_dropout', 0.2),
        'ff_dropout': params.get('ff_dropout', 0.2),
        'explainable': True
    }
    lr = params.get('learning_rate', 1e-4)
    wd = params.get('weight_decay', 1e-5)

    model = FTTransformer(encoder=FTTransformerEncoder(**encoder_params), out_dim=1,
                          depth=params.get('transformer_depth', 6), out_activation='sigmoid')
    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd),
        loss={"output": tf.keras.losses.BinaryCrossentropy(), "importances": None},
        metrics={"output": [tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]}
    )
    return model, tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True), weighted


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


def best_f2_threshold(y_probs, y_true):
    thresholds = np.linspace(0, 1, 100)
    f2_scores = [fbeta_score(y_true, (y_probs >= t).astype(int), beta=2) for t in thresholds]
    best_idx = np.argmax(f2_scores)
    return f2_scores[best_idx], thresholds[best_idx]


def print_metrics_with_ci(**metrics):
    names = {
        'roc_auc': 'ROC AUC Score',
        'pr_auc': 'PR AUC Score',
        'brier': 'Brier Score',
        'f2': 'F2 Score',
        'specificity_list': 'Specificity',
        'sensitivity_list': 'Sensitivity',
        'npv_list': 'Negative Pred Value (NPV)',
        'precision_list': 'Precision (PPV)',
        'recall_list': 'Recall'
    }

    for key, label in names.items():
        mean, ci = calculate_confidence_interval(metrics[key])
        print(f"{label}: {mean:.3f} ({mean - ci:.3f}, {mean + ci:.3f})")

def calculate_confidence_interval(data, confidence=0.95):
    mean, std, n = np.mean(data), statistics.stdev(data), len(data)
    t_val = t.ppf((1 + confidence) / 2., n - 1)
    return mean, t_val * std


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
    classes = ['False', 'True']  #['No delir', 'Delir']
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


def get_shap_importance(X_val, shap_values):
    df_shap_values = pd.DataFrame(data=shap_values, columns=X_val.columns)
    df_feature_importance = pd.DataFrame(columns=['feature', 'importance'])

    for col in df_shap_values.columns:
        importance = df_shap_values[col].abs().mean()
        df_feature_importance.loc[len(df_feature_importance)] = [col, importance]

    return df_feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)


def catboost_fs(clf, X_train, X_test, y_train, y_test, num_features=37):
    """
    Performs feature selection using CatBoost's SHAP-based recursive selection

    Args:
        clf (catboost.CatBoostClassifier): Initialized CatBoost classifier
        X_train (pd.DataFrame): Training feature matrix
        X_test (pd.DataFrame): Testing feature matrix
        y_train (pd.Series): Training target vector
        y_test (pd.Series): Testing target vector
        num_features (int, optional): Number of features to select, 37 results in the lowest loss
    """

    summary = clf.select_features(
        X_train,
        y=y_train,
        eval_set=(X_test, y_test),
        features_for_select=f'0-{len(X_train.columns)-1}',
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


def boruta_feature_selection(X, y, max_iter=100):
    """
    Performs feature selection using the Boruta algorithm with a Random Forest classifier. It identifies 8 all-relevant
    features: 'apachescore', 'age', 'rass_mean', 'temperature_mean', 'gcs_verbal_mean', 'gcs_total_mean',
    'respiratoryrate_mean', 'heartrate_mean'

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix
        y (pd.Series or np.ndarray): Target vector
        max_iter (int, optional): Maximum number of iterations for Boruta

    Returns:
        list: Selected feature names
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced', random_state=1)

    boruta = BorutaPy(rf, verbose=2, random_state=1, max_iter=max_iter, alpha=0.3)
    boruta.fit(X.values, y)

    return X.columns[boruta.support_].tolist()


# Boruta all-relevant feature selection
def borutaShap_feature_selection(X, y, model, n_trials=150):
    """
    Performs Boruta feature selection using importance based on Shapley values. It identifies 11 all-relevant
    features: 'apachescore', 'respiratoryrate_mean', 'gcs_verbal_mean', 'urgentadmission', 'age', 'teachingstatus',
    'first24hrmaxrass', 'heartrate_mean', 'gcs_total_mean', 'first24hrmaxrass', rass_mean

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        model (object): Trained machine learning model compatible with SHAP
        n_trials (int, optional): Number of trials for BorutaShap
    """
    feat_selector = BorutaShap(
        model=model,
        importance_measure='shap',
        classification=True,
    )

    feat_selector.fit(X, y, n_trials=n_trials, random_state=1, verbose=True)
    feat_selector.TentativeRoughFix()
    feat_selector.results_to_csv(filename='borutaShap_importance')
    feat_selector.plot(X_size=12, figsize=(12, 8),
                       y_scale='log', which_features='accepted', display=False)
    plt.savefig('plots/borutaShap.pdf', format="pdf", bbox_inches="tight")


def cv_hpo(X, y, clf, param_space=None, folds=5, n_trials=150):
    """
    Performs hyperparameter optimization using Optuna with cross-validation.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        clf (sklearn.base.BaseEstimator or tuple): Classifier or tuple for feature transformer
        param_space (dict, optional): Hyperparameter search space
        folds (int, optional): Number of cross-validation folds
        n_trials (int, optional): Number of optimization trials

    Returns:
        dict: Best hyperparameters found during optimization
    """
    if param_space is None:
        param_space = {}
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
            print("AUROC: " + str(auroc))
            roc_auc_scores.append(auroc)

        return statistics.mean(roc_auc_scores)

    study.optimize(objective_ft if isinstance(clf, tuple) else objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    return best_params


def train_and_evaluate(X_train, y_train, X_test, y_test, clf, parameters, name):
    """
    Trains and evaluates a classifier using various performance metrics and visualizations.

    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target vector
        X_test (pd.DataFrame): Testing feature matrix
        y_test (pd.Series): Testing target vector
        clf (sklearn.base.BaseEstimator or tuple): Classifier or tuple for feature transformer
        parameters (dict): Model hyperparameters
        name (str): File names

    Returns:
        clf: Trained classifier.
    """
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

    print_metrics_with_ci(
        roc_auc=roc_auc_scores, pr_auc=pr_auc_scores, brier=brier_scores, f2=f2_scores,
        specificity_list=specificity_list, sensitivity_list=sensitivity_list,
        npv_list=npv_list, precision_list=precision_list, recall_list=recall_list
    )

    chance = len(y_test[y_test == 1]) / len(y_test)

    title = 'Non-weighted' if 'notbal' in name else 'Weighted'
    plot_curves(precision_scores, recall_scores, fpr_list, tpr_list, roc_auc_scores, pr_auc_scores, chance, title, name)
    plot_calibration(y_pred_proba, y_test, name)
    best_f2, f2_threshold = best_f2_threshold(y_pred_proba, y_test)
    print("Best F2 score: {:.3f} (at threshold {:.3f})".format(best_f2, f2_threshold))
    plot_confusion_matrix(clf, X_test, y_test, name, normalize=False, title=title)
    if 'ft' not in name:
        get_shap(clf, X_test, name)
    return clf


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    plt.style.use('seaborn-paper')
    matplotlib.rcParams.update({'figure.autolayout': True})

    # Load parameters and feature set from JSON files
    params = load_json('params.json')


    df = prepare_data('data/first24hourdata_new.csv', 'event')
    #df = prepare_data('data/heart_disease.csv', 'HeartDiseaseorAttack')
    #df = prepare_data('data/titanic.csv', 'Survived')

    features = load_json('features.json')
    # smaller_featureset = features['feature_set'] + ['target']
    #df = df[smaller_featureset]
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

    #best_params = cv_hpo(X_train, y_train, ft_notbal_clf, ft_space, folds=4, n_trials=50)

    # Change model and parameters here
    clf = train_and_evaluate(X_train, y_train, X_test, y_test, LR_bal_clf, LR_bal_params, name="LR_bal")

    #borutaShap_feature_selection(X_test, y_test, cb_bal_clf)

    #cb_bal_clf.set_params(**cb_bal_params)
    #catboost_fs(cb_bal_clf, X_train, X_test, y_train, y_test, num_features=5)

    plt.show()


if __name__ == '__main__':
    main()

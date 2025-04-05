import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler


def prepare_data(location="data/first24hourdata_new.csv", target='event'):
    df = pd.read_csv(location)
    df = df.dropna()

    df = df.rename(columns={target: "target"})

    df['target'] = df['target'] == 1

    df.columns = [x.lower() for x in df.columns]
    df = df.drop(
        ['patientunitstayid', 'tstart', 'tend', 'interval', 'unnamed_0', 'dummy_1', 'dummy_2', 'dummy_3', 'delir',
         'offset', 'interval', 'cause'], axis=1, errors='ignore')

    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype('category').cat.codes

    binary_cols = df.columns[(df.nunique() == 2) & df.isin([0, 1]).all()]
    df[binary_cols] = df[binary_cols].astype(bool)

    return df


def show_corr(df, var1, var2):
    if df[var1].dtype == 'object':
        var1, var2 = var2, var1

    sns.set_context("talk", font_scale=1.5)

    if df[var1].dtype in ['int64', 'float64']:
        # Numerical vs Numerical (Scatter plot)
        if df[var2].dtype in ['int64', 'float64']:
            ax = sns.scatterplot(data=df, x=var1, y=var2)
            ax.set_xlabel(var1, fontsize=18)
            ax.set_ylabel(var2, fontsize=18)
            ax.set_title(f'{var1} vs {var2}', fontsize=20)

        # Numerical vs Categorical (Violin plot)
        else:
            ax = sns.violinplot(data=df, x=var2, y=var1, inner="box")
            ax.set_xlabel(var2, fontsize=18)
            ax.set_ylabel(var1, fontsize=18)
            ax.set_title(f'{var1} vs {var2}', fontsize=20)
    else:
        # Categorical vs Numerical (Violin plot)
        if df[var2].dtype in ['int64', 'float64']:
            ax = sns.violinplot(data=df, x=var1, y=var2, inner="box")
            ax.set_xlabel(var1, fontsize=18)
            ax.set_ylabel(var2, fontsize=18)
            ax.set_title(f'{var2} vs {var1}', fontsize=20)

        # Categorical vs Categorical (Pie Charts)
        else:
            crosstab = pd.crosstab(df[var1], df[var2], margins=True)
            print(crosstab)

            counts_true = df[[var1, var2]].where(df[var1]).value_counts().rename_axis([var1, var2]).reset_index(
                name='counts')
            counts_false = df[[var1, var2]].where(df[var1] == False).value_counts().rename_axis(
                [var1, var2]).reset_index(name='counts')

            plt.style.use('seaborn-paper')
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Bigger figure size

            total_true = counts_true['counts'].sum()
            total_false = counts_false['counts'].sum()

            def fmt1(x):
                return '{:.0f}'.format(total_true * x / 100)

            def fmt2(x):
                return '{:.0f}'.format(total_false * x / 100)

            ax[0].pie(counts_true['counts'], autopct=fmt1, textprops={'color': "w", 'fontsize': 22},
                      labels=counts_true[var2], colors=['darkgreen', 'blue', 'grey', 'red'])
            ax[0].set_title(var1 + ' = True', fontsize=24)
            ax[0].text(0.5, -0.05, 'total: ' + str(total_true), transform=ax[0].transAxes, ha="center", fontsize=20,
                       color="gray")

            ax[1].pie(counts_false['counts'], autopct=fmt2, textprops={'color': 'w', 'fontsize': 22},
                      labels=counts_false[var2], colors=['darkgreen', 'blue', 'grey', 'red'])
            ax[1].set_title(var1 + ' = False', fontsize=24)
            ax[1].text(0.5, -0.05, 'total: ' + str(total_false), transform=ax[1].transAxes, ha="center", fontsize=20,
                       color="gray")

            handles, labels = plt.gca().get_legend_handles_labels()
            if df[var2].dtype == bool:
                labels = [var2 + ' = False', var2 + ' = True']
            fig.legend(handles, labels, loc='lower center', fontsize=18)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.tight_layout()
    plt.savefig(f'plots/corr_{var1}_{var2}.pdf', format='pdf')
    plt.show()


def top_corr(df, method, variable=None, drop_outcome=False, top_n=20):
    df = df.copy()
    if drop_outcome:
        corr = df.drop('target', axis=1).corr(method=lambda x, y: method(x, y)[0])
        pvalues = df.drop('target', axis=1).corr(method=lambda x, y: method(x, y)[1]) - np.eye(
            len(df.drop('target', axis=1).columns))
    else:
        corr = df.corr(method=lambda x, y: method(x, y)[0])
        pvalues = df.corr(method=lambda x, y: method(x, y)[1]) - np.eye(len(df.columns))

    corr_long = corr.stack().reset_index()
    corr_long.columns = ['variable_1', 'variable_2', 'correlation']

    pvalues_long = pvalues.stack().reset_index()
    pvalues_long.columns = ['variable_1', 'variable_2', 'p_value']

    merged = corr_long.merge(pvalues_long, on=['variable_1', 'variable_2'])
    merged = merged[merged['variable_1'] != merged['variable_2']]

    merged['pair_key'] = merged.apply(lambda row: tuple(sorted([row['variable_1'], row['variable_2']])), axis=1)
    merged = merged.drop_duplicates(subset=['pair_key']).drop(columns=['pair_key'])

    if variable:
        merged = merged[(merged['variable_1'] == variable) | (merged['variable_2'] == variable)]
        merged['variable'] = np.where(
            merged['variable_1'] == variable,
            merged['variable_2'],
            merged['variable_1']
        )
        merged = merged[['variable', 'correlation', 'p_value']]

    return merged.sort_values(by='correlation', key=abs, ascending=False).reset_index(drop=True, inplace=False).head(
        top_n)


def spearman_corr(df, variable=None, drop_outcome=False, top_n=20):
    return top_corr(df, ss.spearmanr, variable, drop_outcome, top_n)


def pearson_corr(df, variable=None, drop_outcome=False, top_n=20):
    return top_corr(df, ss.pearsonr, variable, drop_outcome, top_n)


def plot_corr(pearson_df, spearman_df):
    merged_df = pearson_df.merge(spearman_df, on="variable", suffixes=("_pearson", "_spearman"))
    merged_df = merged_df.reindex(merged_df["correlation_pearson"].abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(10, 8))

    bar_width = 0.35
    y_pos = np.arange(len(merged_df))

    ax.barh(y_pos - bar_width / 2, merged_df["correlation_pearson"], bar_width, label="Pearson", color="royalblue")
    ax.barh(y_pos + bar_width / 2, merged_df["correlation_spearman"], bar_width, label="Spearman", color="darkorange")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged_df["variable"], fontsize=12)
    ax.set_xlabel("Correlation Coefficient", fontsize=14)
    ax.set_title("Feature Correlations with Outcome Variable", fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.axvline(x=0, color="gray", linestyle="--")

    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig("plots/pearson_spearman_corr.pdf", format="pdf")
    plt.show()


def cramers_v(x, y):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    result = -1
    if len(x.value_counts()) == 1:
        print("First variable is constant")
    elif len(y.value_counts()) == 1:
        print("Second variable is constant")
    else:
        conf_matrix = pd.crosstab(x, y)

        if conf_matrix.shape[0] == 2:
            correct = False
        else:
            correct = True

        chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2 / n

        r, k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        result = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return round(result, 6)


def get_correlation_categorical(df, var1, var2):
    return cramers_v(df[var1], df[var2])


def pie(df):
    plt.style.use('seaborn-paper')
    event_counts = df.groupby('target').size()
    labels = event_counts.index

    colors = plt.cm.Paired(range(len(labels)))

    plt.figure()
    plt.pie(event_counts, labels=labels, autopct='%.2f%%', colors=colors, startangle=140,
            textprops={'fontsize': 12, 'color': 'black'})

    plt.title('Target Distribution', fontsize=16)
    plt.savefig("plots/pie_vis.pdf", format="pdf", tight_layout=True)
    plt.show()


def binarize_outcome_variable(df, outcome_var='event', positive='delirium', negative='no delirium'):
    df = df[df[outcome_var] != 'none'].copy()
    categories = df[outcome_var].unique()

    for category in categories:
        if category not in [positive, negative]:
            df.loc[:, outcome_var] = df[outcome_var].map({category: negative})
    df.loc[:, outcome_var] = df[outcome_var].map({positive: True, negative: False})
    return df


def main():
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams.update({'figure.autolayout': True})
    pd.set_option('display.max_columns', 7)
    plt.style.use('seaborn-paper')

    df = prepare_data('data/heart_disease.csv', 'HeartDiseaseorAttack')

    pie(df)
    print(df.describe())
    print(df.columns.values.tolist())

    pearson_df = pearson_corr(df, 'target')
    spearman_df = spearman_corr(df, 'target')
    plot_corr(pearson_df, spearman_df)

    print(f"Cramers V: {cramers_v(df['fruits'], df['veggies'])}")

    show_corr(df, 'smoker', 'target')
    show_corr(df, 'bmi', 'target')
    show_corr(df, 'bmi', 'menthlth')

    # sns.pairplot(df[['age', 'apachescore', 'event']], hue='event')


if __name__ == '__main__':
    main()

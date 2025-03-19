import pandas as pd
import numpy as np
import plotly.express as px
from seaborn import violinplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler


def get_data():
    df = pd.read_csv('data/first24hourdata_new.csv')
    df.columns = [x.lower() for x in df.columns]
    df = df.drop(
        ['patientunitstayid', 'tstart', 'tend', 'interval', 'unnamed_0', 'dummy_1', 'dummy_2', 'dummy_3', 'delir',
         'offset', 'interval', 'cause'], axis=1, errors='ignore')

    # Identify binary columns
    binary_cols = df.columns[(df.nunique() == 2) & df.isin([0, 1]).all()]
    df[binary_cols] = df[binary_cols].astype(bool)

    df['event'] = df['event'].map({0.0: "none", 1.0: "delirium", 2.0: "death", 3.0: "release"})
    return df


def show_corr(df, var1, var2):
    plt.figure(figsize=(8, 6))

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

            ax[0].pie(counts_true['counts'], autopct=fmt1, textprops={'color': "w", 'fontsize': 14},
                      labels=counts_true[var2], colors=['darkgreen', 'blue', 'grey', 'red'])
            ax[0].set_title(var1, fontsize=20)
            ax[0].text(0.5, -0.05, 'total: ' + str(total_true), transform=ax[0].transAxes, ha="center", fontsize=16,
                       color="gray")

            ax[1].pie(counts_false['counts'], autopct=fmt2, textprops={'color': 'w', 'fontsize': 14},
                      labels=counts_false[var2], colors=['darkgreen', 'blue', 'grey', 'red'])
            ax[1].set_title('no ' + var1, fontsize=20)
            ax[1].text(0.5, -0.05, 'total: ' + str(total_false), transform=ax[1].transAxes, ha="center", fontsize=16,
                       color="gray")

            handles, labels = plt.gca().get_legend_handles_labels()
            if df[var2].dtype == bool:
                labels = [var2 + ' = False', var2 + ' = True']
            fig.legend(handles, labels, loc='lower center', fontsize=16)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.tight_layout()
    plt.savefig(f'plots/corr_{var1}_{var2}.pdf', format='pdf')

def top_corr(df, method, variable=None, top_n=20):
    if variable != 'event':
        corr = df.drop('event', axis=1).corr(method=lambda x, y: method(x, y)[0])
        pvalues = df.drop('event', axis=1).corr(method=lambda x, y: method(x, y)[1]) - np.eye(
            len(df.drop('event', axis=1).columns))
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
        merged = merged[
            (merged['variable_1'] == variable) | (merged['variable_2'] == variable)]
        merged['variable'] = merged.apply(
            lambda row: row['variable_2'] if row['variable_1'] == variable else row['variable_1'], axis=1)
        merged = merged[['variable', 'correlation', 'p_value']]

    return merged.sort_values(by='correlation', key=abs, ascending=False).reset_index(drop=True, inplace=False).head(
        top_n)


def spearman_corr(df, variable=None, top_n=20):
    return top_corr(df, ss.spearmanr, variable, top_n)

def pearson_corr(df, variable=None, top_n=20):
    return top_corr(df, ss.pearsonr, variable, top_n)


def plot_corr(pearson_df, spearman_df):
    feature_names = [
        "Mean GCS (total)", "Mean GCS (verbal)", "Mean GCS (eyes)", "First 24hr min RASS", "APACHE IV score",
        "Mean Rass", "Mean GCS (motor)", "Coma", "Ventilated", "Vasopressors", "x24hr metabolicacidosis",
        "Adrenergic bronchodilators", "x24hr maximum temp", "Sedatives hypnotics anxiolytics", "ond",
        "Mean heart rate", "Opioids", "Antibiotics", "Neuromuscular blockers", "Hospital Teaching status"
    ]

    merged_df = pearson_df.merge(spearman_df, on="variable", suffixes=("_pearson", "_spearman"))

    variable_mapping = dict(zip(merged_df["variable"], feature_names))
    merged_df["feature_name"] = merged_df["variable"].map(variable_mapping)

    merged_df = merged_df.dropna(subset=["feature_name"])

    merged_df = merged_df.reindex(merged_df["correlation_pearson"].abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(10, 8))

    bar_width = 0.35
    y_pos = np.arange(len(merged_df))

    ax.barh(y_pos - bar_width / 2, merged_df["correlation_pearson"], bar_width, label="Pearson", color="royalblue")
    ax.barh(y_pos + bar_width / 2, merged_df["correlation_spearman"], bar_width, label="Spearman", color="darkorange")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged_df["feature_name"])
    ax.set_xlabel("Correlation Coefficient")
    ax.set_title("Feature Correlations with Outcome Variable")
    ax.axvline(x=0, color="gray", linestyle="--")  # Reference line at zero

    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/pearson_spearman_corr.pdf", format="pdf")


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


def tsne_vis(df):
    # X = df.drop(['ventilated', 'event'], axis=1)
    X = df[['ventilated', 'rass_mean', 'gcs_total_mean', 'gcs_verbal_mean', 'gcs_eyes_mean', 'apachescore', 'coma',
            'vasopressors']]
    y = df['event']
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    print(tsne_results)

    fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], color=y)
    fig.update_layout(
        title="t-SNE visualization of Custom Classification dataset",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
    )
    fig.show()
    return fig


def pca_vis(df):
    X = df.drop(['event'], axis=1)

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(X)

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df[['event']]], axis=1)
    finalDf = finalDf.dropna()
    print(finalDf)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = ['delirium', 'no delirium']

    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['event'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=20)
    ax.legend(targets)
    ax.grid()
    plt.savefig("plots/pca_vis.pdf", format="pdf", tight_layout=True)


def pie(df, predictor='event'):
    plt.style.use('seaborn-paper')
    event_counts = df.groupby(predictor).size()
    labels = event_counts.index

    colors = plt.cm.Paired(range(len(labels)))

    plt.figure()
    plt.pie(event_counts, labels=labels, autopct='%.2f%%', colors=colors, startangle=140,
            textprops={'fontsize': 12, 'color': 'black'})

    plt.title('Event Distribution', fontsize=16)
    plt.savefig("plots/pie_vis.pdf", format="pdf", tight_layout=True)

def binarize_outcome_variable(df):
    df = df[df['event'] != 'none'].copy()
    df.loc[:, 'event'] = df['event'].map({'delirium': 'delirium', 'death': 'no delirium', 'release': 'no delirium'})
    df.loc[:, 'event'] = df['event'].map({'delirium': True, 'no delirium': False})
    return df

def main():
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams.update({'figure.autolayout': True})
    pd.set_option('display.max_columns', 7)
    plt.style.use('seaborn-paper')

    df = get_data()
    print(df.describe())

    df = binarize_outcome_variable(df)

    pearson_df = pearson_corr(df, 'event')
    spearman_df = spearman_corr(df, 'event')

    plot_corr(pearson_df, spearman_df)

    show_corr(df, 'event', 'admissionheight')

    # print(get_highest_corr(df, 'event', threshold=0.0).head(50))
    # sns.pairplot(df[['age', 'apachescore', 'event']], hue='event')
    plt.show()


if __name__ == '__main__':
    main()

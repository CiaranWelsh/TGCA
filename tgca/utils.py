import os

import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score

from tgca import *


def do_pca(data):
    p = PCA(n_components=2)
    x = p.fit_transform(data.values.transpose())
    var = p.explained_variance_ratio_
    x = pd.DataFrame(x, index=data.columns)
    fig = plt.figure()
    plt.scatter(x[0], x[1])
    sns.despine(fig=fig, top=True, right=True)

    plt.figure()
    y = x[(x[0] < 20000) & (x[1] < 80000)]
    plt.scatter(y[0], y[1])
    plt.xlabel(round(var[0], 3) * 100)
    plt.ylabel(round(var[1], 3) * 100)
    idmax = [1148, 769, 395, 623, 907, 1081]
    plt.scatter(y.loc[idmax, 0], y.loc[idmax, 1])
    plt.show()
    return list(y.index)


def do_umap_by_genes(data, use_pickle=False, gene_subset=[],
                     mpl=True, ply=True, pickle_file=UMAP_BY_GENE_PICKLE):
    if os.path.isfile(pickle_file) and use_pickle:
        embedding = pd.read_pickle(pickle_file)
    else:
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data)
    embedding = pd.DataFrame(embedding, index=data.index)
    embedding.to_pickle(pickle_file)
    subset = embedding[embedding.index.get_level_values(0).isin(gene_subset)]
    # print(subset)

    if mpl:
        fig = plt.figure()
        plt.scatter(embedding[0], embedding[1], marker='.')
        plt.scatter(subset[0], subset[1], label='subset', marker='.')
        sns.despine(fig=fig, top=True, right=True)
        plt.title('UMAP by genes')
        plt.show()
    if ply:
        embedding.columns = ['0', '1']
        embedding.index = data.index
        embedding = embedding.reset_index(level='symbol')
        print(embedding.head())
        # print(data.head())
        import plotly.express as px
        fig = px.scatter(embedding, x='0', y='1',
                         template='plotly_white', hover_data=['symbol'])
        fig.show()
    return embedding


def do_umap_by_sample(data, clusters=[], use_pickle=False, pickle_file=UMAP_BY_SAMPLE_PICKLE):
    if os.path.isfile(pickle_file) and use_pickle:
        embedding = pd.read_pickle(pickle_file)
    else:
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data.transpose())
        embedding = pd.DataFrame(embedding, index=data.columns)
    print('embeddings')
    print(embedding)
    embedding.to_pickle(pickle_file)
    # if clusters != []:
    # num_unique_clusters = clusters['cluster'].unique()
    plt.title('UMAP by samples')
    fig = plt.figure()
    plt.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], marker='.')
    sns.despine(fig=fig, top=True, right=True)
    plt.title('UMAP by sample')
    plt.legend()
    plt.show()
    # print(embedding)
    # import plotly.express as px
    # embedding.columns = ['0', '1']
    # fig = px.scatter(embedding, x='0', y='1')
    #                  hover_data=['petal_width'])
    # fig.show()


def get_clusterable_umap_embedding(
        data, n_neighbors=30,
        min_dist=0.0,
        n_components=50,
        min_samples=3,
        min_cluster_size=3):
    from hdbscan import HDBSCAN
    df = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
    ).fit_transform(data)
    df = pd.DataFrame(df, index=data.index)
    print(df.head())
    labels = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    ).fit_predict(df)
    df['cluster'] = labels
    print(sorted(list(set(labels))))
    fig = plt.figure()
    for label, i in df.groupby(by='cluster', axis=0):
        plt.scatter(
            i.iloc[0, :],
            i.iloc[1, :],
            label=label,
            marker='.'
        )
        sns.despine(fig=fig, top=True, right=True)
    plt.show()
    return df


def get_similarity_measure(data, from_pickle=False, pickle_file=SIMILARITY_MEASURE_PICKLE):
    if from_pickle and os.path.isfile(pickle_file):
        return pd.read_pickle(pickle_file)
    else:
        N = data.shape[1]
        res = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    res[i, j] = np.nan
                else:
                    res[i, j] = ((data.values[:, i] - data.values[:, j]) ** 2).sum()
        res = pd.DataFrame(res)
        res = res / res.max().max()
        res.index = data.columns
        res.columns = data.columns

        res.to_pickle(path=pickle_file)
    return res


def do_scatter(data, x, y):
    fig = plt.figure()
    plt.scatter(data[x], data[y])
    sns.despine(fig=fig, top=True, right=True)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def compute_mi_matrix(data, from_pickle=False, bins=20,
                      pickle_file=MUTUAL_INFORMATION_PICKLE):
    if from_pickle and os.path.isfile(pickle_file):
        return pd.read_pickle(pickle_file)
    N = data.shape[1]
    count = 0
    res = np.zeros(shape=(N, N))
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            count += 1
            print('calculating MI score: {:2f}% completed'.format(count / (N * N)))
            x = data.iloc[:, i]
            y = data.iloc[:, j]
            res[i, j] = calc_MI(x, y, bins=bins)
    df = pd.DataFrame(res, index=data.columns, columns=data.columns)
    df.to_pickle(pickle_file)

    return df

def compute_ks_matrix(
        data, from_pickle=False,
        pickle_file=PROTEOME_KS_PICKLE):
    import scipy.stats as stats
    if from_pickle and os.path.isfile(pickle_file):
        return pd.read_pickle(pickle_file)
    N = data.shape[1]
    count = 0
    ks = np.zeros(shape=(N, N))
    p_val = np.zeros(shape=(N, N))
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            count += 1
            print('calculating MI score: {:2f}% completed'.format(count / (N * N)))
            x = data.iloc[:, i]
            y = data.iloc[:, j]
            k, p = stats.ks_2samp(x.values, y.values)
            ks[i, j] = k
            p_val[i, j] = p
    ks = pd.DataFrame(ks, index=data.columns, columns=data.columns)
    p_val = pd.DataFrame(p_val, index=data.columns, columns=data.columns)
    df = pd.concat({'ks': ks, 'pval': p_val}, axis=0)
    df.to_pickle(pickle_file)

    return df

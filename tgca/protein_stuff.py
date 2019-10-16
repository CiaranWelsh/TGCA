import os

from tgca import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tgca.utils import *

sns.set_context('talk')


def get_data():
    data = pd.concat([pd.read_csv(i) for i in PROTEOME_FILES_LEVEL4], axis=0, sort=False)
    data = data.drop(['Cancer_Type', 'SetID'], axis=1)
    data = data.set_index(['Sample_ID', 'Sample_Type'], append=True)
    data = data[sorted(data.columns)]
    data = data.dropna(how='any', axis=1)
    return data


def see_dist(data):
    fig = plt.figure()
    d = data.iloc[:, 1]
    sns.distplot(d, )
    sns.despine(fig=fig, top=True, right=True)
    plt.show()


if __name__ == '__main__':
    data = get_data()
    print(data)

    n_neighbors = 30
    min_dist = 0.0
    n_components = 50
    min_samples = 3
    min_cluster_size = 3

    # for i in [30, 50, 75, 100, 150]:
    for i in [2, 5, 10, 30, 50, 75, 100, 150]:
        em = get_clusterable_umap_embedding(
            data,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=i,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,

        )
    # print(data.var())
    # do_umap_by_sample(data, pickle_file=UMAP_BY_INDIVIDUAL_PICKLE)
    # do_umap_by_genes(data,
    #                  pickle_file=UMAP_BY_PROTEIN_PICKLE,
    #                  ply=False)

    # see_dist(data)

    # mi = compute_mi_matrix(
    #     data, from_pickle=True,
    #     pickle_file=PROTEOME_MI_PICKLE,
    #     bins=50
    # )
    #
    # fig = plt.figure()
    # sns.clustermap(mi)
    #
    # plt.show()

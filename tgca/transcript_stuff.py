from tgca import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mygene
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

from tgca.utils import *


def map_ids(ids=[], from_pickle=False):
    if not isinstance(ids, list):
        raise ValueError
    if os.path.isfile(ID_MAP_PICKLE) and from_pickle:
        return pd.read_pickle(ID_MAP_PICKLE)

    mg = mygene.MyGeneInfo()
    out = mg.querymany(ids, scopes='ensembl.gene', fields='symbol', species='human')
    ids = pd.DataFrame(out)[['query', 'symbol']]
    ids.to_pickle(ID_MAP_PICKLE)
    return ids


def get_data(from_pickle=False):
    if os.path.isfile(PANDAS_PICKLE) and from_pickle is True:
        return pd.read_pickle(PANDAS_PICKLE)
    gz = [i for i in glob.glob(TRANSCRIPT_DATA_DIRECTORY + '/*/*.gz')]
    data = []
    for num, i in enumerate(gz):
        print('reading {} of {}'.format(num, len(gz)))
        data.append(pd.read_csv(i, sep='\t', header=None))
    data = [i.set_index(0).sort_index() for i in data]

    print('concatonating data')
    df = pd.concat(data, axis=1, sort=False)
    df.columns = range(len(df.columns))

    # replace 0.0 with nan and count
    df['nacount'] = df.replace(0.0, np.nan).isna().sum(axis=1)

    # drop rows with more than 10% nan
    df = df[df['nacount'] < df.shape[0] / 10.0]

    # impute remaining with median
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df)
    df = pd.DataFrame(imputer.transform(df), columns=df.columns, index=df.index)
    df['nacount'] = df.isna().sum(axis=1)
    assert all(df['nacount'] == 0)
    df = df.drop('nacount', axis=1)
    # print(df.shape)
    #
    # df.to_csv(os.path.join(DATA_DIRECTORY, 'data.csv'))

    df['var'] = df.var(axis=1)
    df = df.sort_values(by='var', ascending=True)
    df['var_rank'] = range(df.shape[0])
    df.drop('var', inplace=True, axis=1)
    # print(df.shape)

    df['gene_id'], df['gene_num'] = zip(*[i.split('.') for i in df.index])
    genes = list(df['gene_id'].unique())
    map = map_ids(genes, from_pickle=True)
    df = df.merge(map, left_on='gene_id', right_on='query')
    df = df.set_index(['symbol', 'gene_id', 'var_rank', 'gene_num'], drop=True).drop('query', axis=1)
    # print(df.shape)
    print(df.shape)
    #

    df.to_pickle(PANDAS_PICKLE)
    return df


def vln(data, genes, ncols=3, filename=None, showextrema=False, **kwargs):
    # work out nrows and cols
    nplots = len(genes)
    if nplots == 1:
        ncols = 1
    nrows = int(nplots / ncols)
    remainder = nplots % ncols
    if remainder > 0:
        nrows += 1

    fig = plt.figure()
    for i, gene in enumerate(genes):
        if gene not in data.index.get_level_values('symbol'):
            print('gene "{}" not in data'.format(gene))
            continue
        plot_data = data.xs(gene, level='symbol')
        ax = plt.subplot(nrows, ncols, i + 1)
        plt.violinplot(plot_data, showextrema=showextrema, **kwargs)
        sns.despine(ax=ax, top=True, right=True)
        plt.xlabel('Amount (FKPM)')
        plt.ylabel('Frequency')
        plt.title(gene)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight')


def do_tsne(data):
    tsne = TSNE()


def get_genes_from_kegg_pathway(pathway):
    from bioservices.kegg import KEGG
    k = KEGG()
    k.organism = 'hsa'
    pathway = k.get(pathway)
    genes = k.parse(pathway)['GENE']
    entrez, symbol = zip(*[i.split('  ') for i in genes])
    return symbol


def read_in_clinical_data():
    import json
    with open(CLINICAL_FILE) as f:
        data = json.load(f)
    # print(data[0].keys())
    # print(data[0]['diagnoses'])
    return data


def do_hist(data):
    fig = plt.figure()
    plt.scatter(range(len(data.index)), data[151])
    plt.show()
    # print(data)


# @jit(nopython=True)


# def compare_two


def kmeans(data, nclusters=3):
    from sklearn.cluster import KMeans
    k = KMeans(n_clusters=nclusters)
    data = data.transpose()
    print(data)
    k.fit(data.values)
    print(k.inertia_)
    print(k.labels_)
    # for i in range(nclusters):








if __name__ == '__main__':
    data = get_data(from_pickle=True)
    genes = list(set(data.index.get_level_values(0)))
    map = map_ids(from_pickle=True)

    # looks like we have some fairly extreme values.
    # Are they always the same samples? If so -> artifact. If not -> Interesting.

    # removes outliers determined from pca
    # keep_idx = do_pca(data)
    # data = data[keep_idx]

    # reorder by variance (since we've removed outliers, need to recomputer)
    # data['var'] = data.var(axis=1)
    # print(data[data['var'] > ])
    # sns.distplot(data[0])
    from sklearn.preprocessing import Normalizer
    scaler = Normalizer()
    data = pd.DataFrame(scaler.fit_transform(data.transpose()),
                        columns=data.index, index=data.columns).T
    data['var'] = data.var(axis=1)
    data = data.sort_values(by='var', ascending=False).drop('var', axis=1)

    PERCENT = 5
    data = data.iloc[:data.shape[0]//PERCENT, :]

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


    # do_umap_by_genes(data, use_pickle=True, mpl=False)
    # do_umap_by_sample(data, use_pickle=False)
    # data = data[data['var'] > 5]

    # genes = list(set(data.index.get_level_values(1)))
    # print(genes)
    #
    # with open(os.path.join(DATA_DIRECTORY, 'variable_gene_list.csv'), 'w') as f:
    #     for i in genes:
    #         f.write(i + '\n')
    #
    #
    #
    # sns.distplot(data[4])
    # plt.show()
    # fig = plt.figure()
    # plt.plot(range(len(variance)), variance.values / variance.max())
    # plt.show()

    # kmeans(data)

    # similarity = get_similarity_measure(data, from_pickle=True)
    # print(similarity)
    # print(similarity.idxmax(axis=1).unique())
    # idmax = [1148, 769, 395, 623, 907, 1081]

    # print(data.head())
    # print(data[idmax])
    # print(similarity[idmax])
    # print(similarity[idmax].sum())

    # mi = compute_mi_matrix(data, from_pickle=True)
    # print(mi)
    # fig = plt.figure()
    # cl = sns.clustermap(mi)
    #
    # from scipy.cluster.hierarchy import fcluster
    #
    # for i in sorted(dir(cl)):
    #     print(i)
    # fl = fcluster(cl.dendrogram_row.linkage, 3, criterion='maxclust')
    # clusters = pd.DataFrame({j: i for i, j in zip(fl, list(cl.data2d))}, index=['cluster']).transpose()

    # fig = plt.figure()
    # sns.clustermap(similarity.fillna(0))
    # plt.show()
    #
    # ones = clusters[clusters['cluster'] == 1]
    # twos = clusters[clusters['cluster'] == 2]
    # threes = clusters[clusters['cluster'] == 3]

    # print(data)
    # ones = data[list(ones.index)]
    # twos = data[list(twos.index)]
    # threes = data[list(threes.index)]

    # print(ones)

    # do_umap_by_sample(data, clusters, use_pickle=True)

    # for i in idmax:
    #     for j in idmax:
    #         do_scatter(data, i, j)

    # print(similarity.idxmax())
    # print(data[1130])
    # similarity = similarity.stack()
    # x = similarity.idxmin().reset_index()
    # print(x.head())
    # print(similarity.head())

    # print(similarity.iloc[x[0]])
    # print(similarity.min().min())
    # print(similarity)

    # print(similarity.min())

    # print(similarity.idxmin())

    # print(similarity.idxmin(axis=1))
    # similarity.to_csv(SIMILARITY_DATA_CSV)
    # data.to_csv(DATA_CSV_FILE)
    # do_hist(data)

    # read_in_clinical_data()
    # mapk_genes = get_mapk_genes()
    # pi3k_genes = get_genes_from_kegg_pathway('hsa04151')
    # mapk_genes = get_genes_from_kegg_pathway('hsa04010')
    # do_umap_by_genes(data, use_pickle=True, gene_subset=mapk_genes)
    # do_pca(data)
    # print(data.loc['IDO1'].values)

    # vln(data, ['IDO1', 'KYNU'], ncols=2)
    # vln(data, ['IDO1', 'TMEM220'], ncols=2)
    # plt.figure()
    # plt.scatter(x=range(len(data.loc['IDO1'].values[0])), y=data.loc['IDO1'].values[0])
    #
    # plt.show()

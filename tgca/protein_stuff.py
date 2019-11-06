import os

from tgca import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tgca.utils import *

sns.set_context('talk')


def biology_led_subset():
    """
    List of proteins that are part of PI3K/MAPK systems
    :return:
    """
    return [
        'AKT', 'AKT_pS473', 'AKT_pT308', 'AMPKALPHA', 'AMPKALPHA_pT172',
        'BRCA2', 'CJUN_pS73', 'EGFR', 'EGFR_pY1068', 'EGFR_pY1173', 'ERALPHA',
        'ERALPHA_pS118', 'ERK2', 'ETS1', 'FOXO3A', 'FOXO3A_pS318S321', 'GSK3ALPHABETA',
        'GSK3ALPHABETA_pS21S9', 'GSK3_pS9', 'HER2', 'HER2_pY1248', 'HER3', 'HER3_pY1289',
        'IGF1R_pY1135Y1136', 'IGFBP2', 'IRS1', 'JAK2', 'JNK2', 'JNK_pT183Y185', 'LKB1',
        'MAPK_pT202Y204', 'MEK1', 'MEK1_pS217S221', 'MTOR', 'MTOR_pS2448', 'NFKBP65_pS536',
        'P16INK4A', 'P38MAPK', 'P38_pT180Y182', 'P53', 'P70S6K1', 'P70S6K_pT389', 'P90RSK',
        'P90RSK_pT359S363', 'PDK1', 'PDK1_pS241', 'PI3KP110ALPHA', 'PI3KP85', 'PKCALPHA',
        'PKCALPHA_pS657', 'PKCDELTA_pS664', 'PRAS40_pT246', 'PTEN', 'RAPTOR', 'RICTOR',
        'RICTOR_pT1135', 'S6', 'S6_pS235S236', 'S6_pS240S244', 'SHC_pY317', 'SHP2_pY542',
        'SRC', 'SRC_pY416', 'SRC_pY527', 'STAT3_pY705', 'TSC1', 'TUBERIN', 'TUBERIN_pT1462',
        'VEGFR2']


def get_data():
    data = pd.concat([pd.read_csv(i) for i in PROTEOME_DATA_FILE], axis=0, sort=False)
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


def plot_histograms(data, subset):
    for i in subset:
        d = data[i]
        fig = plt.figure()
        plt.hist(d, bins=50)
        sns.despine(fig=fig, top=True, right=True)
        plt.title('{} (n={})'.format(i, d.shape[0]))
        fname = os.path.join(LEVEL4_PROTEOME_DIR_HISTOGRAMS, f'hist_{i}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')


def qq_plot(data, subset):
    import scipy.stats as stats

    for i in subset:
        d = data[i]
        fig = plt.figure()
        stats.probplot(d, dist='norm', plot=plt)
        sns.despine(fig, top=True, right=True)
        plt.title(i)
        fname = os.path.join(LEVEL4_PROTEOME_DIR_HISTOGRAMS, f'qq_{i}.png')
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print('saved to {}'.format(fname))
        if not os.path.isfile(fname):
            raise FileNotFoundError(fname)


# def kstest(data, subset):
#     import scipy.stats as stats
#     res = {}
#     for i in subset[:2]:
#         d = data[i]
#         print(stats.kstest(d.values, 'norm'))
# res[i] = stats.kstat(d.values, 'norm')
# return res

if __name__ == '__main__':
    data = get_data()
    # print(data)

    n_neighbors = 30
    min_dist = 0.0
    n_components = 50
    min_samples = 3
    min_cluster_size = 3

    all_measured_proteins = sorted(['ACC1', 'ACC_pS79', 'ACETYLATUBULINLYS40', 'ACVRL1', 'ADAR1',
                                    'AKT', 'AKT_pS473', 'AKT_pT308', 'AMPKALPHA', 'AMPKALPHA_pT172',
                                    'ANNEXIN1', 'ANNEXINVII', 'AR', 'ARAF', 'ARAF_pS299', 'ARID1A',
                                    'ASNS', 'ATM', 'BAD_pS112', 'BAK', 'BAP1C4', 'BAX', 'BCL2', 'BCL2A1',
                                    'BCLXL', 'BECLIN', 'BETACATENIN', 'BID', 'BIM', 'BRAF', 'BRAF_pS445',
                                    'BRCA2', 'BRD4', 'CABL', 'CASPASE3', 'CASPASE7CLEAVEDD198', 'CASPASE8',
                                    'CAVEOLIN1', 'CD20', 'CD26', 'CD31', 'CD49B', 'CDK1', 'CDK1_pY15', 'CHK1',
                                    'CHK1_pS296', 'CHK1_pS345', 'CHK2', 'CHK2_pT68', 'CIAP', 'CJUN_pS73', 'CKIT',
                                    'CLAUDIN7', 'CMET', 'CMET_pY1235', 'CMYC', 'COG3', 'COLLAGENVI', 'CRAF',
                                    'CRAF_pS338', 'CYCLINB1', 'CYCLIND1', 'CYCLINE1', 'CYCLINE2', 'DIRAS3',
                                    'DJ1', 'DUSP4', 'DVL3', 'ECADHERIN', 'EEF2', 'EEF2K', 'EGFR', 'EGFR_pY1068',
                                    'EGFR_pY1173', 'EIF4E', 'EIF4G', 'EPPK1', 'ERALPHA', 'ERALPHA_pS118', 'ERCC1',
                                    'ERCC5', 'ERK2', 'ETS1', 'FASN', 'FIBRONECTIN', 'FOXM1', 'FOXO3A',
                                    'FOXO3A_pS318S321',
                                    'G6PD', 'GAB2', 'GAPDH', 'GATA3', 'GSK3ALPHABETA', 'GSK3ALPHABETA_pS21S9',
                                    'GSK3_pS9',
                                    'HER2', 'HER2_pY1248', 'HER3', 'HER3_pY1289', 'HEREGULIN', 'HSP70',
                                    'IGF1R_pY1135Y1136',
                                    'IGFBP2', 'INPP4B', 'IRF1', 'IRS1', 'JAB1', 'JAK2', 'JNK2', 'JNK_pT183Y185', 'KU80',
                                    'LCK', 'LKB1', 'MAPK_pT202Y204', 'MEK1', 'MEK1_pS217S221', 'MIG6', 'MRE11', 'MSH2',
                                    'MSH6', 'MTOR', 'MTOR_pS2448', 'MYH11', 'MYOSINIIA_pS1943', 'NCADHERIN',
                                    'NDRG1_pT346',
                                    'NF2', 'NFKBP65_pS536', 'NOTCH1', 'NRAS', 'P16INK4A', 'P21', 'P27', 'P27_pT157',
                                    'P27_pT198', 'P38MAPK', 'P38_pT180Y182', 'P53', 'P62LCKLIGAND', 'P70S6K1',
                                    'P70S6K_pT389',
                                    'P90RSK', 'P90RSK_pT359S363', 'PAI1', 'PARPCLEAVED', 'PAXILLIN', 'PCADHERIN',
                                    'PCNA',
                                    'PDCD4', 'PDK1', 'PDK1_pS241', 'PDL1', 'PEA15', 'PEA15_pS116', 'PI3KP110ALPHA',
                                    'PI3KP85',
                                    'PKCALPHA', 'PKCALPHA_pS657', 'PKCDELTA_pS664', 'PKCPANBETAII_pS660', 'PR',
                                    'PRAS40_pT246',
                                    'PRDX1', 'PREX1', 'PTEN', 'RAB11', 'RAB25', 'RAD50', 'RAD51', 'RAPTOR', 'RB',
                                    'RBM15',
                                    'RB_pS807S811', 'RICTOR', 'RICTOR_pT1135', 'S6', 'S6_pS235S236', 'S6_pS240S244',
                                    'SCD1',
                                    'SETD2', 'SF2', 'SHC_pY317', 'SHP2_pY542', 'SMAC', 'SMAD1', 'SMAD3', 'SMAD4',
                                    'SNAIL', 'SRC',
                                    'SRC_pY416', 'SRC_pY527', 'STAT3_pY705', 'STAT5ALPHA', 'STATHMIN', 'SYK', 'TAZ',
                                    'TFRC', 'TIGAR',
                                    'TRANSGLUTAMINASE', 'TSC1', 'TUBERIN', 'TUBERIN_pT1462', 'VEGFR2', 'X1433BETA',
                                    'X1433EPSILON',
                                    'X1433ZETA', 'X4EBP1', 'X4EBP1_pS65', 'X4EBP1_pT37T46', 'X4EBP1_pT70', 'X53BP1',
                                    'XBP1', 'XRCC1',
                                    'YAP', 'YAP_pS127', 'YB1', 'YB1_pS102'])

    res = compute_ks_matrix(data, from_pickle=True)

    ks = pd.DataFrame(res.loc['ks'].stack())
    ks = ks.replace(0.0, np.nan).dropna(how='any')

    # fig = plt.figure()
    # sns.distplot(ks.values)
    # plt.show()
    # print(ks)

    ks_shortlist = ks[ks[0] < 0.065]
    print(ks_shortlist)

    pairs = zip(ks_shortlist.index.get_level_values(0),
                ks_shortlist.index.get_level_values(1),
                ks_shortlist.values)

    fname = os.path.join(LEVEL4_PROTEOME_DIR, 'ks_stats.csv')
    ks.to_csv(fname)

    # import networkx as nx
    #
    # G = nx.Graph()
    # for i, j, v in pairs:
    #     G.add_edge(i, j, ks=v)
    #
    # print(G)
    #
    # fig = plt.figure()
    # nx.draw_networkx_labels(G, pos=nx.spring_layout(G),
    #                         node_size=10, node_color='black')
    # plt.show()
    #
    # print(ks_shortlist)

    # print(ks.describe())
    # fig = plt.figure()
    # sns.clustermap(data=ks[ks[0] < 0.05])
    # plt.show()
    # plot_histograms(data, list(data.columns))
    # qq_plot(data, list(data.columns))
    # res = kstest(data, list(data.columns))
    # for k, v in res.items():
    #     print(k, v)

    # for i in [30, 50, 75, 100, 150]:
    # for i in [2, 5, 10, 30, 50, 75, 100, 150]:
    #     em = get_clusterable_umap_embedding(
    #         data,
    #         n_neighbors=n_neighbors,
    #         min_dist=min_dist,
    #         n_components=i,
    #         min_samples=min_samples,
    #         min_cluster_size=min_cluster_size,
    #
    #     )
    # print(data)
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

import pandas as pd
import numpy as np
import os, glob


import sompy

from tgca import *
import random
import joblib

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
    data = pd.concat([pd.read_csv(i) for i in PROTEOME_FILES_LEVEL4], axis=0, sort=False)
    data = data.drop(['Cancer_Type', 'SetID'], axis=1)
    data = data.set_index(['Sample_ID', 'Sample_Type'], append=True)
    data = data[sorted(data.columns)]
    data = data.dropna(how='any', axis=1)
    return data

def do_self_organising_map(data):
    mapsize = [20, 20]
    som = sompy.SOMFactory.build(data, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var',
                                 initialization='pca', neighborhood='gaussian', training='batch', name='sompy')
    som.train(n_job=1, verbose=None)
    v = sompy.mapview.View2DPacked(50, 50, 'test', text_size=8)
    # could be done in a one-liner: sompy.mapview.View2DPacked(300, 300, 'test').show(som)
    v.show(som, what='codebook', which_dim=[0, 1], cmap=None, col_sz=6)  # which_dim='all' default
    # v.save('2d_packed_test')

if __name__ == '__main__':
    data = get_data()
    sample_ids = data.index
    col_ids = data.columns


    data = data.reset_index(drop=True)
    data.columns = range(data.shape[1])
    print(data)
    som = do_self_organising_map(data)
    print(som)




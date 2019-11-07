import os
import glob
import subprocess
try:
    # for when working on the cluster
    subprocess.check_call('sinfo')
    WORKING_DIRECTORY = '/mnt/nfs/home/b7053098/ciaran/TGCA'
except FileNotFoundError:
    # for when working locally
    WORKING_DIRECTORY = '/media/ncw135/DATA/TGCA'

TGCA_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'tgca')
DATA_DIRECTORY = os.path.join(TGCA_DIRECTORY, 'data')
TRANSCRIPT_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'transcriptome_FPKM')
DATA_CSV_FILE = os.path.join(DATA_DIRECTORY, 'transcriptome_data.csv')
ID_MAP_PICKLE = os.path.join(DATA_DIRECTORY, 'id_map.pickle')
UMAP_BY_GENE_PICKLE = os.path.join(DATA_DIRECTORY, 'umap_by_gene.pickle')
UMAP_BY_SAMPLE_PICKLE = os.path.join(DATA_DIRECTORY, 'umap_by_sample.pickle')
PANDAS_PICKLE = os.path.join(DATA_DIRECTORY, 'data.pickle')
CLINICAL_FILE = os.path.join(DATA_DIRECTORY, 'clinical.cart.2019-10-14.json')
SIMILARITY_MEASURE_PICKLE = os.path.join(DATA_DIRECTORY, 'similarity_measure.pickle')
SIMILARITY_DATA_CSV = os.path.join(DATA_DIRECTORY, 'similarity_measure.csv')
MUTUAL_INFORMATION_PICKLE = os.path.join(DATA_DIRECTORY, 'mutual_infomation.pickle')

PROTEOME_DATA_DIR = os.path.join(DATA_DIRECTORY, 'proteome')
LEVEL3_PROTEOME_DIR = os.path.join(PROTEOME_DATA_DIR, 'level3')
LEVEL4_PROTEOME_DIR = os.path.join(PROTEOME_DATA_DIR, 'level4')
PROTEOME_DATA_FILE = os.path.join(LEVEL4_PROTEOME_DIR, 'TCGA-BRCA-L4.csv')

PROTEOME_MI_PICKLE = os.path.join(LEVEL4_PROTEOME_DIR, 'MI.pickle')
PROTEOME_KS_PICKLE = os.path.join(LEVEL4_PROTEOME_DIR, 'KS.pickle')
UMAP_BY_PROTEIN_PICKLE = os.path.join(DATA_DIRECTORY, 'umap_by_gene.pickle')
UMAP_BY_INDIVIDUAL_PICKLE = os.path.join(DATA_DIRECTORY, 'umap_by_sample.pickle')

LEVEL4_PROTEOME_DIR_HISTOGRAMS = os.path.join(LEVEL4_PROTEOME_DIR, 'histograms')

GENETIC_ALGORITHM_DATA_DIR = os.path.join(DATA_DIRECTORY, 'genetic_algorithm_data')
GENETIC_ALGORITHM_PICKLES_DIR = os.path.join(GENETIC_ALGORITHM_DATA_DIR, 'genetic_algorithm_pickles')
GENETIC_ALGORITHM_PCA_PLOTS_DIR = os.path.join(GENETIC_ALGORITHM_DATA_DIR, 'PCAPlots')
GENETIC_ALGORITHM_UMAP_PLOTS_DIR = os.path.join(GENETIC_ALGORITHM_DATA_DIR, 'UMAPPlots')
GENETIC_ALGORITHM_TSNE_PLOTS_DIR = os.path.join(GENETIC_ALGORITHM_DATA_DIR, 'TSNEPlots')
GENETIC_ALGORITHM_DIAGNOSIS_PLOTS_DIR = os.path.join(GENETIC_ALGORITHM_DATA_DIR, 'Diagnosis')
for i in [GENETIC_ALGORITHM_DATA_DIR, GENETIC_ALGORITHM_PICKLES_DIR,
          GENETIC_ALGORITHM_PCA_PLOTS_DIR, GENETIC_ALGORITHM_TSNE_PLOTS_DIR,
          GENETIC_ALGORITHM_UMAP_PLOTS_DIR, GENETIC_ALGORITHM_DIAGNOSIS_PLOTS_DIR]:
    if not os.path.isdir(i):
        os.makedirs(i)

GENETIC_ALGORITHM_RESULTS_PICKLES = glob.glob(os.path.join(GENETIC_ALGORITHM_DATA_DIR, '*.pickle'))

# todo: Feature extraction
# todo: Try SVM/Random forest/SOM unsupervised version




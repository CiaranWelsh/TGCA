
import os
import glob

WORKING_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
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
PROTEOME_FILES_LEVEL3 = glob.glob(os.path.join(LEVEL3_PROTEOME_DIR, '*/*/*.csv'))
PROTEOME_FILES_LEVEL4 = glob.glob(os.path.join(LEVEL4_PROTEOME_DIR, '*/*/*.csv'))
PROTEOME_MI_PICKLE = os.path.join(LEVEL4_PROTEOME_DIR, 'MI.pickle')
UMAP_BY_PROTEIN_PICKLE = os.path.join(DATA_DIRECTORY, 'umap_by_gene.pickle')
UMAP_BY_INDIVIDUAL_PICKLE = os.path.join(DATA_DIRECTORY, 'umap_by_sample.pickle')













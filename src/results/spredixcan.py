import os
import re
from glob import glob
from collections import defaultdict

import numpy as np
from scipy import stats
import pandas as pd

from utils import simplify_string_for_hdf5
import settings as conf
from entity import Trait


class PhenoInfo:
    def __init__(self, directory_path):
        if not os.path.isdir(directory_path):
            raise ValueError(f'Directory does not exist: {directory_path}')

        self.pheno_path = directory_path

        base_dir_name = os.path.basename(self.pheno_path)
        self.pheno_code = base_dir_name

        self.trait = Trait(self.pheno_code)

        self.pheno_description = self.trait.description

    def get_plain_name(self):
        return self.trait.get_plain_name()

    def __str__(self):
        if not pd.isnull(self.pheno_description):
            return f'{self.pheno_code} - {self.pheno_description}'
        else:
            return self.pheno_code

    def __repr__(self):
        return self.__str__()


class PhenoResults:
    DEFAULT_FILE_PATTERN = re.compile('(?P<code>.+)-gtex_v8-(?P<tissue>.+)-(?P<date>.+)\.csv$')

    def __init__(self, pheno_info, file_pattern=None):
        if isinstance(pheno_info, PhenoInfo):
            self.pheno_info = pheno_info
        else:
            self.pheno_info = PhenoInfo(pheno_info)

        if file_pattern is None:
            self.file_pattern = PhenoResults.DEFAULT_FILE_PATTERN
        else:
            self.file_pattern = re.compile(file_pattern)

        self._init_metadata()

    def _init_metadata(self):
        self.csv_files = []
        csv_files_matches = []
        for csv_f in glob(os.path.join(self.pheno_info.pheno_path, '*')):
            mat = re.search(self.file_pattern, csv_f)
            if mat is None:
                continue

            self.csv_files.append(csv_f)
            csv_files_matches.append(mat)

        self.file_by_tissue = {m.group('tissue'): m.string for m in csv_files_matches}
        self.tissues = [m.group('tissue') for m in csv_files_matches]

        # hdf5
        hdf5_pattern = re.compile('spredixcan-(?P<tissue>.+)-(?P<column>.+)\.h5')
        hdf5_files_matches = []
        for hdf5_f in glob(os.path.join(conf.GENE_ASSOC_DIR, 'spredixcan', '*.h5')):
            mat = re.search(hdf5_pattern, hdf5_f)
            if mat is None:
                continue

            hdf5_files_matches.append(mat)

        self.hdf5_files = defaultdict(dict)
        for m in hdf5_files_matches:
            self.hdf5_files[m.group('tissue')][m.group('column')] = m.string

    def get_consensus_effect_direction(self, pval_threshold=1e-4):
        def _get_effect_direction(zscores):
            zscores = zscores.dropna()
            if zscores.shape[0] == 0:
                return np.nan

            pvalues = pd.Series(stats.norm.cdf(np.abs(zscores) * -1.0) * 2.0, index=zscores.index.tolist())

            pvalues = pvalues[pvalues < pval_threshold]
            if pvalues.shape[0] == 0:
                return np.nan

            zscores = zscores.loc[pvalues.index]
            zscores_sign = np.sign(zscores)
            zscores_sign_counts = zscores_sign.value_counts()
            # check if there is a tie
            if zscores_sign_counts.shape[0] == 2 and zscores_sign_counts.unique().shape[0] == 1:
                return 0.0
            return zscores_sign_counts.sort_values(ascending=False).index[0]

        data_dict = {t:self.get_tissue_data(t, 'zscore', index_col='gene_simple') for t in self.tissues}
        data = pd.DataFrame(data_dict)
        return data.apply(_get_effect_direction, axis=1)

    def get_most_significant_effect_direction(self):
        def _get_effect_direction(zscores):
            zscores = zscores.dropna()
            if zscores.shape[0] == 0:
                return np.nan

            x_min, x_max = zscores.min(), zscores.max()

            # if min and max are similar, then it's 0 (tie)
            if np.isclose(np.abs(x_min), np.abs(x_max)):
                return 0.0

            max_abs_zscores = max(x_min, x_max, key=abs)
            return np.sign(max_abs_zscores)

        data_dict = {t:self.get_tissue_data(t, 'zscore', index_col='gene_simple') for t in self.tissues}
        data = pd.DataFrame(data_dict)
        return data.apply(_get_effect_direction, axis=1)

    def get_tissue_data(self, tissue, cols, index_col='gene'):
        if tissue not in self.tissues:
            return None

        if not isinstance(cols, (tuple, list)):
            cols = [cols]

        squeeze = len(cols) == 1

        #tissue_file_path = self.file_by_tissue[tissue]
        #df = pd.read_csv(tissue_file_path, usecols=['gene'] + cols)

        # FIXME: only the first column specified is read
        hdf5_tissue_file_path = self.hdf5_files[tissue][cols[0]]
        with pd.HDFStore(hdf5_tissue_file_path, mode='r') as store:
            clean_col = simplify_string_for_hdf5(self.pheno_info.get_plain_name())
            df = store[clean_col]

        #if index_col == 'gene_simple':
        #    df = df.assign(gene_simple=df['gene'].apply(lambda x: x.split('.')[0]))
        #    df = df.drop(columns=['gene'])

        #df = df.set_index(index_col)
        if squeeze:
            df = df.squeeze()

        if isinstance(df, pd.Series):
            df = df.rename(self.pheno_info.get_plain_name())

        assert df.index.is_unique

        return df

    def __str__(self):
        return 'S-PrediXcan results for ' + self.pheno_info.__str__()

    def __repr__(self):
        return self.__str__()

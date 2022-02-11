'''
Plot scatterplots: behavioral data exp 1 vs low-level feaures

python figures/figure6b1.py \
    --task jip_janneke \
    --beh_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp2_29subjs  \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --n_folds 12
'''

import sys
sys.path.insert(0, '.')
import argparse
import os.path as op
import pandas as pd
import seaborn as sns
import copy

from scipy import stats
from matplotlib import pyplot as plt
from figures.figure4b import load_pearson_df
from figures.figure4c import load_vad_df
from figures.figure4d import load_stoi_df
from utils.private.datasets import get_subjects_by_code
from figures.figure6a1 import plot_scatter_beh_metric
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(args):
    res_beh = pd.read_csv(op.join(args.beh_dir, 'results_avg_over_subjs.csv'), index_col=[0])
    res_beh = res_beh[res_beh['recon']==True].sort_values(by=['sub', 'mod', 'word']).reset_index(drop=True)

    for metric in args.metric:
        args_ext = copy.copy(args)
        args_ext.metric = metric
        if metric == 'pearsonr':
            res_low = load_pearson_df(args_ext)
        elif metric == 'vad':
            res_low = load_vad_df(args_ext)
            metric = 'vad_match'
        elif metric == 'stoi':
            res_low = load_stoi_df(args_ext)
        else:
            raise ValueError

        for isub, subject in enumerate(args.subject):
            res_low.loc[res_low['subject']==subject, 'subject'] = isub + 1
        res_low['subject'] = res_low['subject'].astype(int)

        res_low = res_low[res_low['trial']=='optimized'].sort_values(by=['subject', 'model', 'word']).reset_index(drop=True)

        if metric == 'stoi':
            assert(res_low.shape[0] == res_beh.shape[0])
            filter = res_low[res_low[metric] == 1e-5].index
        else:
            filter = []

        for a, b in zip(['sub', 'mod', 'word'], ['subject', 'model', 'word']):
            assert res_beh.drop(filter)[a].equals(res_low.drop(filter)[b])

        plot_scatter_beh_metric(res_beh.drop(filter), res_low.drop(filter), metric,
                                title='fig6b1_beh_exp2_speeaker_id_' + metric + '_scatter',
                                plotdir=args.plot_dir)


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot exp 1')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--beh_dir',  type=str, help='Path to behavioral results')
    parser.add_argument('--res_dir',  type=str, help='Path to metric results')
    parser.add_argument('--plot_dir', type=str, help='Path to plot')
    parser.add_argument('--metric', '-x', type=str, nargs="+",
                        choices=['pearsonr', 'vad', 'stoi'],
                        default=['pearsonr', 'vad', 'stoi'],
                        help='Metric to use')
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Model to run')
    parser.add_argument('--type_perm',  type=str,
                        choices=['shift', 'shuffle'],
                        default='shift',
                        help='Model to run')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=True)
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)



'''
Plot scatterplots: behavioral data exp 1 vs low-level feaures

python figures/figure6a1.py \
    --task jip_janneke \
    --metric pearsonr \
    --beh_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp1_30subjs \
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

from scipy import stats
from matplotlib import pyplot as plt
from utils.general import get_stat_pval
from figures.figure4b import load_pearson_df
from utils.private.datasets import get_subjects_by_code

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(args):
    res_beh = pd.read_csv(op.join(args.beh_dir, 'results_avg_over_30subjs.csv'), index_col=[0])
    res_low = load_pearson_df(args)
    res_beh = res_beh[res_beh['recon']==True]
    res_low = res_low[res_low['trial']=='optimized']

    metric = args.metric

    for model in args.model:
        fig, ax = plt.subplots(1, 5)
        plt.title(model)
        a, b = [], []
        for isub, subject in enumerate(args.subject):
            # plt.scatter(res_beh[res_beh['mod']==model]['accuracy'], res_low[res_low['model']==model][metric])
            sns.regplot(x=res_beh[(res_beh['mod']==model)&(res_beh['sub']==isub+1)]['accuracy'],
                        y=res_low[(res_low['model']==model)&(res_low['subject']==subject)][metric], ax=ax[isub])
            a.append(res_beh[(res_beh['mod']==model)&(res_beh['sub']==isub+1)]['accuracy'].median())
            b.append(res_low[(res_low['model']==model)&(res_low['subject']==subject)][metric].median()[0])
        plt.figure()
        sns.regplot(x=a, y=b)
        slope, intercept, r_value, p_value, std_err = stats.linregress(a,b)
        print(slope, p_value)
        plt.title(model)

##
    colors = sns.color_palette()[:3]
    for imodel, model in enumerate(args.model):
        slope, intercept, r_value, p_value, std_err = stats.linregress(res_beh[(res_beh['mod']==model)]['accuracy'].values,
                                                                       res_low[(res_low['model']==model)][metric].values.flatten())

        plt.figure()
        plt.title(model)
        sns.regplot(x=res_beh[(res_beh['mod']==model)]['accuracy'],
                    y=res_low[(res_low['model']==model)][metric],
                    color=colors[imodel],
                    line_kws={'label':"y={0:.2f}x+{1:.2f},r={2:.2f},p={3:.5f}".format(slope,intercept, r_value, p_value)})
        plt.legend()
        print(slope, p_value)
        if args.plot_dir is not None:
            plt.savefig(op.join(args.plot_dir, title + '.pdf'), dpi=160, transparent=True)
            plt.close()



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



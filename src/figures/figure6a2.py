'''
Plot scatterplots: behavioral data exp 1 vs low-level feaures

python figures/figure6a2.py \
    --task jip_janneke \
    --beh_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp1_30subjs \
    --gen_res_dir /Fridge/users/julia/project_decoding_jip_janneke/results \
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
from figures.figure5a import load_classify_df
from utils.private.datasets import get_subjects_by_code

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plot_scatter_beh_metric(res_beh, res_low, metric, title='', plotdir=None):
    slope, intercept, r_value, p_value, std_err = stats.linregress(res_beh['accuracy'].values,
                                                                   res_low[metric].values.flatten())

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.regplot(x=res_beh['accuracy'],
                y=res_low[metric],
                line_kws={'label':"y={0:.2f}x+{1:.2f},r={2:.2f},p={3:.5f}".format(slope,intercept, r_value, p_value)},
                ax=ax)
    plt.legend()
    print(slope, p_value)

    markers = ['*', '^', 'o', 's', 'd']
    colors = sns.color_palette()[:3]
    n = len(res_beh['sub'].unique())
    pos = ax.get_children()[0].get_offsets().data
    c = 0
    for sub, mar in zip(range(n), markers):
        for model in range(3):
            marsize = 11 if mar == '*' else 7  # was 11 and 15: too big
            ax.plot(pos[c, 0], pos[c, 1], markersize=marsize,
                    markeredgecolor='black', color=colors[model], marker=mar)
            c += 1

    if plotdir is not None:
        plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
        plt.close()


def main(args):
    res_beh = pd.read_csv(op.join(args.beh_dir, 'results_avg_over_subjs.csv'), index_col=[0])
    res_beh = res_beh[res_beh['recon']==True].sort_values(by=['sub', 'mod', 'word']).reset_index(drop=True)

    args_ext = copy.copy(args)
    res_low = load_classify_df(args_ext)

    for isub, subject in enumerate(args.subject):
        res_low.loc[res_low['subject']==subject, 'subject'] = isub + 1
    res_low['subject'] = res_low['subject'].astype(int)

    res_low = res_low[(res_low['trial']=='optimized')&(res_low['input']=='reconstructed')].sort_values(by=['subject']).reset_index(drop=True)
    res_beh = res_beh.groupby(['sub', 'mod']).mean().reset_index()
    res_beh = res_beh.iloc[res_beh['mod'].map({'mlp':0, 'densenet':1, 'seq2seq':2}).sort_values().index].sort_values(by='sub').reset_index(drop=True)

    for a, b in zip(['sub', 'mod'], ['subject', 'model']):
        assert res_beh[a].equals(res_low[b])

    plot_scatter_beh_metric(res_beh, res_low, 'accuracy',
                            title='fig6a2_beh_exp1_word_id_' + args.metric + '_scatter',
                            plotdir=args.plot_dir)


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot exp 1')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--beh_dir',  type=str, help='Path to behavioral results')
    parser.add_argument('--gen_res_dir',  type=str, help='Path to metric results')
    parser.add_argument('--plot_dir', type=str, help='Path to plot')
    parser.add_argument('--metric', '-x', type=str,
                        choices=['classify'],
                        default='classify',
                        help='Metric to use')
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Model to run')
    parser.add_argument('--clf_type', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='logreg',
                        help='Type of classifier')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=True)
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)



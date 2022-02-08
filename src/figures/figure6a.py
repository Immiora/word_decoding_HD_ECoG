'''
Plot behavioral data exp 1
Boxplot over participants

python figures/figure6a.py \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp1_30subjs \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures
'''

import sys
sys.path.insert(0, '.')
import argparse
import os.path as op
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from utils.general import get_stat_pval

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def plot_box_acc(data, floors=None, title='', plot_dots=False, plotdir=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    sns.boxplot(x='sub', y='accuracy', data=data[data['recon'] == True], hue='mod', hue_order=['mlp', 'densenet', 'seq2seq'],
                boxprops=dict(alpha=.7),
                medianprops=dict(color='red'),
                ax=ax, orient='v', showfliers=False, whis=[5, 95])
    sns.boxplot(x='sub', y='accuracy', data=data[data['recon'] == False], hue='mod',
                boxprops=dict(alpha=.3, linewidth=.5, facecolor='salmon'), orient='v',
                whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                capprops=dict(linestyle=':', linewidth=.5, color='black'),
                ax=ax, color='gray', showfliers=False, whis=[5, 95])

    if floors is not None:
        sns.boxplot(x='sub', y='accuracy', data=floors[floors['recon'] == True], hue='mod', hue_order=['mlp', 'densenet', 'seq2seq'],
                    boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
                    whiskerprops = dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    ax=ax, orient='v',
                    showfliers=False, color='gray', whis=[5, 95])

    if plot_dots:
        sns.stripplot(x='sub', y='accuracy', data=data[data['recon'] == True],
                      hue='mod', size=6, orient='v', hue_order=['mlp', 'densenet', 'seq2seq'],
                      jitter=.2, dodge=True, alpha=.4, linewidth=.6,
                      edgecolors='black', ax=ax)
        # sns.stripplot(x='sub', y='accuracy', data=data[data['recon'] == False],
        #               hue='mod', size=4, orient='v', jitter=True, dodge=True, color='grey', alpha=0.3, ax=ax)
        # if floors is not None:
        #     sns.stripplot(x='sub', y='accuracy', data=floors[floors['recon'] == True],
        #                   hue='mod', size=4, orient='v', jitter=True, dodge=True, color='grey', alpha=0.3, ax=ax)

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if plotdir is not None:
        plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
        plt.close()



def main(args):
    res = pd.read_csv(op.join(args.res_dir, 'results_avg_over_12words.csv'), index_col=[0])
    res_perm = pd.read_csv(op.join(args.res_dir, 'results_avg_over_12words_avg_over_subjs_perm.csv'), index_col=[0])
    plot_box_acc(res, res_perm, title='fig6a_beh_exp1_word_id', plot_dots=args.plot_dots, plotdir=args.plot_dir)

    for s in [i for i in range(1, 6)]:
        for m in ['mlp', 'densenet', 'seq2seq']:
            baseline = res_perm[(res_perm['sub'] == s) & (res_perm['mod'] == m)]['accuracy'].values
            val = res[(res['sub'] == s) & (res['recon'] == True) & (res['mod'] == m)]['accuracy'].median()
            print(str(s) + ' & ' + m + ' pval: ' + str(get_stat_pval(val, baseline)))


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot exp 1')
    parser.add_argument('--res_dir',  type=str, help='Path to results')
    parser.add_argument('--plot_dir', type=str, help='Path to plot')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=True)
    args = parser.parse_args()
    main(args)

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
import numpy as np

from matplotlib import pyplot as plt
from utils.general import get_stat_pval

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def plot_box_acc(data, floors=None, ceilings=None, title='', ylim = (0, 1), plotdir=None):
    fig, ax = plt.subplots(1, 1, figsize=(4, 5)) # changed from 5,3 to 6,3

    if floors is not None:
        sns.boxplot(y='accuracy', data=floors,
                    orient='v',
                    boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    showfliers=False, whis=[5, 95], ax=ax)

    if ceilings is not None:
        sns.boxplot(y='accuracy', data=ceilings,
                    orient='v',
                    boxprops=dict(alpha=.3, linewidth=.5, facecolor='salmon'),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    ax=ax, color='gray', showfliers=False, whis=[5, 95])

    sns.stripplot(x='mod', y='accuracy', data=data,
                  size=9, orient='v',
                  jitter=.2, dodge=True, alpha=.0, linewidth=.6,
                  edgecolors='black', ax=ax)

    sns.boxplot(x='mod', y='accuracy', data=data,
                boxprops=dict(alpha=.7), width=.7,
                medianprops=dict(color='red'),
                ax=ax, orient='v', showfliers=False, whis=[5, 95])

    markers = ['*', '^', 'o', 's', 'd']
    colors = sns.color_palette()[:3]
    n = len(data['sub'].unique())
    m = len(data['file'].unique())
    # m = len(data['word'].unique())
    for box in range(3):
        print(box)
        pos = ax.get_children()[box].get_offsets().data
        print(len(pos))
        for sub, mar in zip(range(n), markers):
            marsize = 11 if mar == '*' else 7  # was 11 and 15: too big
            ax.plot(np.nanmean(pos[:, 0])+0.16*(sub-2),
                    np.nanmedian(pos[sub*m:sub*m+m, 1]), markersize=marsize+4,
                    markeredgecolor='black', color=colors[box], marker=mar)
            for i in range(m): # replace with unique folds?
                ax.plot(np.mean(pos[:, 0])+0.16*(sub-2), pos[sub*m+i][1], markersize=marsize,
                        markeredgecolor='black', color=colors[box], marker=mar, alpha=0.1)

    plt.ylim(ylim)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if plotdir is not None:
        plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
        plt.close()

def main(args):
    res = pd.read_csv(op.join(args.res_dir, 'results_avg_over_12words.csv'), index_col=[0])
    # res = pd.read_csv(op.join(args.res_dir, 'results_avg_over_subjs.csv'), index_col=[0])
    res_perm = pd.read_csv(op.join(args.res_dir, 'results_avg_over_12words_avg_over_subjs_perm.csv'), index_col=[0])

    for s in [i for i in range(1, 6)]:
        for m in ['mlp', 'densenet', 'seq2seq']:
            baseline = res_perm[(res_perm['sub'] == s) & (res_perm['mod'] == m)]['accuracy'].values
            val = res[(res['sub'] == s) & (res['recon'] == True) & (res['mod'] == m)]['accuracy'].median()
            print(str(s) + ' & ' + m + ' pval: ' + str(get_stat_pval(val, baseline)))

    # model effects
    from scipy import stats
    import scikit_posthocs as spost
    a = res[(res['recon'] == True) & (res['mod'] == 'mlp')]['accuracy']
    b = res[(res['recon'] == True) & (res['mod'] == 'densenet')]['accuracy']
    c = res[(res['recon'] == True) & (res['mod'] == 'seq2seq')]['accuracy']
    print(stats.kruskal(a, b, c))
    print(spost.posthoc_dunn(np.vstack([a, b, c])))


    for i, m in enumerate(['mlp', 'densenet', 'seq2seq']):
        res.loc[res['mod']==m, 'mod'] = str(i) + '_' + m
    res = res.sort_values(by=['sub', 'mod'])

    plot_box_acc(res[res['recon'] == True], floors=res_perm, ceilings=res[res['recon'] == False],
                 title='fig6a_beh_exp1_word_id_sub_medians',
                 ylim=(.05, 1.05),
                 plotdir=args.plot_dir)




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

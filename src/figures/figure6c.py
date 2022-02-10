'''
Plot behavioral data exp 3
Boxplot over participants

python figures/figure6c.py \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp3_29subjs \
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
from figures.figure6a import plot_box_acc

# def plot_box_acc(data, floors=None, title='', plot_dots=False, plotdir=None):
#     fig, ax = plt.subplots(1, 1, figsize=(6, 3))
#     sns.boxplot(x='sub', y='weight', data=data, hue='mod',
#                 boxprops=dict(alpha=.7), medianprops=dict(color='red'),
#                 hue_order=['mlp', 'densenet', 'seq2seq'],
#                 ax=ax, orient='v', showfliers=False, whis=[5, 95])
#     # sns.boxplot(x='sub', y='weights', data=data, hue='mod', boxprops=dict(alpha=.3), orient='v',
#     #             ax=ax, color='gray', showfliers=False, whis=[5, 95])
#
#     if floors is not None:
#         sns.boxplot(x='sub', y='weight', data=floors, hue='mod', hue_order=['mlp', 'densenet', 'seq2seq'],
#                     boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
#                     whiskerprops = dict(linestyle=':', linewidth=.5, color='black'),
#                     medianprops=dict(linestyle=':', linewidth=.5, color='black'),
#                     capprops=dict(linestyle=':', linewidth=.5, color='black'),
#                     ax=ax, orient='v',
#                     showfliers=False, color='gray', whis=[5, 95])
#
#     if plot_dots:
#         sns.stripplot(x='sub', y='weight', data=data,
#                       hue='mod', size=6, orient='v', hue_order=['mlp', 'densenet', 'seq2seq'],
#                       jitter=.2, dodge=True, alpha=.4, linewidth=.6,
#                       edgecolors='black', ax=ax)
#         # sns.stripplot(x='sub', y='accuracy', data=data,
#         #               hue='mod', size=10, orient='v', jitter=0, dodge=True, color='grey', alpha=0.3, ax=ax)
#
#     plt.title(title)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     if plotdir is not None:
#         plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
#         plt.close()



def main(args):
    res = pd.read_csv(op.join(args.res_dir, 'results_avg_over_words_opt.csv'), index_col=[0])
    res_perm = pd.read_csv(op.join(args.res_dir, 'results_avg_over_words_avg_over_subjs_opt_perm.csv'), index_col=[0])


    for s in [i for i in range(1, 6)]:
        for m in ['mlp', 'densenet', 'seq2seq']:
            baseline = res_perm[(res_perm['sub'] == s) & (res_perm['mod'] == m)]['weight'].values
            val = res[(res['sub'] == s) & (res['mod'] == m)]['weight'].median()
            print(str(s) + ' & ' + m + ' pval: ' + str(get_stat_pval(val, baseline)))

    for i, m in enumerate(['mlp', 'densenet', 'seq2seq']):
        res.loc[res['mod']==m, 'mod'] = str(i) + '_' + m
    res = res.sort_values(by=['sub', 'mod'])

    res = res.rename(columns={'weight':'accuracy'})
    res_perm = res_perm.rename(columns={'weight':'accuracy'})
    plot_box_acc(res, floors=res_perm,
                 title='fig6c_beh_exp3_optimized',
                 ylim=(-.075, 0.105),
                 plotdir=args.plot_dir)
    #plot_box_acc(res, res_perm, title='fig6c_beh_exp3_optimized', plot_dots=args.plot_dots, plotdir=args.plot_dir)

##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot exp 3')
    parser.add_argument('--res_dir',  type=str, help='Path to results')
    parser.add_argument('--plot_dir', type=str, help='Path to plot')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=True)
    args = parser.parse_args()
    main(args)



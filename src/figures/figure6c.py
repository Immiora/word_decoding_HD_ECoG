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
from numpy import vstack

def main(args):
    res = pd.read_csv(op.join(args.res_dir, 'results_avg_over_words_opt.csv'), index_col=[0])
    res_perm = pd.read_csv(op.join(args.res_dir, 'results_avg_over_words_avg_over_subjs_opt_perm.csv'), index_col=[0])

    for s in [i for i in range(1, 6)]:
        for m in ['mlp', 'densenet', 'seq2seq']:
            baseline = res_perm[(res_perm['sub'] == s) & (res_perm['mod'] == m)]['weight'].values
            val = res[(res['sub'] == s) & (res['mod'] == m)]['weight'].median()
            print(str(s) + ' & ' + m + ' pval: ' + str(get_stat_pval(val, baseline)))

    # model effects
    from scipy import stats
    import scikit_posthocs as spost
    a = res[(res['mod'] == 'mlp')]['weight']
    b = res[(res['mod'] == 'densenet')]['weight']
    c = res[(res['mod'] == 'seq2seq')]['weight']
    print(stats.kruskal(a, b, c))
    print(spost.posthoc_dunn(vstack([a, b, c])))


    for i, m in enumerate(['mlp', 'densenet', 'seq2seq']):
        res.loc[res['mod']==m, 'mod'] = str(i) + '_' + m
    res = res.sort_values(by=['sub', 'mod'])

    res = res.rename(columns={'weight':'accuracy'})
    res_perm = res_perm.rename(columns={'weight':'accuracy'})
    plot_box_acc(res, floors=res_perm,
                 title='fig6c_beh_exp3_optimized_sub_medians',
                 ylim=(-.075, 0.105),
                 plotdir=args.plot_dir)

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



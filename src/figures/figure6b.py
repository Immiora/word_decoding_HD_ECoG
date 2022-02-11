'''
Plot behavioral data exp 2
Boxplot over participants

python figures/figure6b.py \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp2_29subjs \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures
'''

import sys
sys.path.insert(0, '.')
import argparse
import os.path as op
import pandas as pd

from figures.figure6a import plot_box_acc
from utils.general import get_stat_pval
from numpy import vstack

def main(args):
    res = pd.read_csv(op.join(args.res_dir, 'results_avg_over_12words.csv'), index_col=[0])
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
    print(spost.posthoc_dunn(vstack([a, b, c])))


    for i, m in enumerate(['mlp', 'densenet', 'seq2seq']):
        res.loc[res['mod']==m, 'mod'] = str(i) + '_' + m
    res = res.sort_values(by=['sub', 'mod'])

    plot_box_acc(res[res['recon'] == True], floors=res_perm, ceilings=res[res['recon'] == False],
                 title='fig6b_beh_exp2_word_id_sub_medians',
                 ylim=(.05, 1.05),
                 plotdir=args.plot_dir)




##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot exp 2')
    parser.add_argument('--res_dir',  type=str, help='Path to results')
    parser.add_argument('--plot_dir', type=str, help='Path to plot')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=True)
    args = parser.parse_args()
    main(args)



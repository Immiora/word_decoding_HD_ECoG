'''
plot speaker classification results: from audio reconstructions and audio data with permutation baseline

python figures/figure5d1.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --clf_type logreg \
    --n_folds 12
'''

import sys
sys.path.insert(0, '.')

import pandas as pd
import argparse
import os.path as op
import seaborn as sns
import numpy as np

from os import makedirs
from utils.general import get_stat_pval
from matplotlib import pyplot as plt
#from figures.figure5a import plot_classify_boxplot

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(args):

    if not op.exists(args.plot_dir): makedirs(args.plot_dir)

    for metric in args.metric:
        results = pd.DataFrame()
        gen_dir = op.join(args.res_dir, args.task)

        for input in ['reconstructed', 'target_audio']: # have to keep outside otherwise in plots: mlp, brain12, densenet, seq2seq

            for model in args.model:

                for itrial, trial in enumerate(['trial0_', '']):
                    if input == 'reconstructed' or (input == 'target_audio' and trial != 'trial0_'):
                        a = pd.read_csv(op.join(gen_dir, 'eval_' + model + '_' + metric + '_' + trial + args.clf_type + '.csv'), index_col=[0])
                        b = {}
                        b['input'] = input
                        b['accuracy'] = a[a['input'] == input]['accuracy'].mean()
                        b['model'] = [model if input == 'reconstructed' else 'audio']
                        b['trial'] = 'non-optimized' if trial == 'trial0_' else 'optimized'
                        b['case'] = args.clf_type
                        results = results.append(pd.DataFrame(b))

                # UNCOMMENT TO ADD PERM
                a = pd.read_csv(op.join(gen_dir, 'eval_' + model + '_' + metric + '_' + args.clf_type + '_perm.csv'), index_col=[0])
                b = {}
                for perm in range(1000):
                    b['input'] = input
                    #b['accuracy'] = a[a['input'] == input]['accuracy'].sample(n=100).mean()
                    b['accuracy'] = a[(a['input'] == input) & (a['perm'] == perm)]['accuracy'].mean() # already means, but keep this for consistency
                    b['model'] = [model if input == 'reconstructed' else 'audio']
                    b['trial'] = 'permutation'
                    b['case'] = args.clf_type
                    results = results.append(pd.DataFrame(b))

        #
        # results.loc[results['trial'] == 'non-optimized', 'input'] = 'non-optimized'

        fig, ax = plt.subplots(1, 1, figsize=(2, 4))
        sns.stripplot(x='input', y='accuracy', data=results[(results['trial'].isin(['optimized', 'non-optimized']))&(results['input']=='reconstructed')],
                      hue='model', hue_order=['mlp', 'densenet', 'seq2seq'], size=9, orient='v',
                      jitter=.2, dodge=True, alpha=.4, linewidth=.6, linestyle='-',
                      edgecolors='black', ax=ax)

        # ceiling: based on audio
        sns.boxplot(y='accuracy', data=results[(results['trial']=='optimized')&(results['input']=='target_audio')],
                    hue='model',
                    orient='v',
                    boxprops=dict(alpha=.3, linewidth=.5, facecolor='salmon'),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    ax=ax, color='gray', showfliers=False, whis=[5, 95])

        # floors: based on permutation
        # sample 10,000 from all three models (densenet, mlp, seq2seq), average, plot one distribution
        temp = results[(results['trial']=='permutation')&(results['input']=='reconstructed')].copy()
        x = []
        for model in args.model:
            x.append(temp[temp['model']==model]['accuracy'].sample(10000, replace=True).values)
        x = np.array(x)
        x = np.mean(x,0)

        sns.boxplot(y='accuracy', data=pd.DataFrame({'accuracy':x}),
                    boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    orient='v',
                    ax=ax, showfliers=False, whis=[5, 95])  # added showfliers = False, whis=[5, 95]

        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if args.plot_dir is not None:
            plt.savefig(op.join(args.plot_dir,
                                     'fig5d1_eval_' + metric + '_' + args.clf_type + '_recon_audio' + '_opt_non-opt.pdf'),
                        dpi=160, transparent=True)
            plt.close()

        for model in args.model:
            for input in ['reconstructed']:
                for trial in ['non-optimized', 'optimized']:
                    baseline = results[(results['case']==args.clf_type) & (results['input']==input) &
                                            (results['model']==model) &
                                            (results['trial']=='permutation')]['accuracy'].values
                    val = results[(results['case'] == args.clf_type) & (results['input']==input) &
                                       (results['trial']==trial) &
                                       (results['model'] == model)]['accuracy'].mean()
                    print(args.clf_type + ' & ' + model +
                                          ' & ' + input +
                                          ' & ' + trial + ' pval: ' + str(get_stat_pval(val, baseline)))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Subject to run')
    parser.add_argument('--clf_feat', '-f', type=str,  nargs="+",
                        choices=['default', 'avg_time', 'avg_chan'],
                        default=['default', 'avg_time', 'avg_chan'],
                        help='Input classifier features')
    parser.add_argument('--metric', '-x', type=str,  nargs="+",
                        choices=['classify_speakers'],
                        default=['classify_speakers'],
                        help='Metric to use')
    parser.add_argument('--clf_type', '-c', type=str,
                        choices=['svm', 'mlp', 'logreg'],
                        default=['svm', 'mlp', 'logreg'],
                        help='Type of classifier')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=False)

    args = parser.parse_args()

    main(args)

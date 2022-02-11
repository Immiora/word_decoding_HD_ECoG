'''
plot word classification results: from audio reconstructions and raw brain data with permutation baseline

python figures/figure5a.py \
    --task jip_janneke \
    --gen_res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/ \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --clf_type logreg \
    --n_folds 12
'''

import sys
sys.path.insert(0, '.')

import pandas as pd
import json
import argparse
import seaborn as sns
import os.path as op

from scipy.stats import ttest_rel
from utils.wilcoxon import wilcoxon
from matplotlib import pyplot as plt
from os import makedirs
from utils.private.datasets import get_subjects_by_code

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_classify_df(args):
    if not op.exists(args.plot_dir): makedirs(args.plot_dir)

    results = pd.DataFrame()
    for subject in args.subject:
        print(subject)
        for input in ['reconstructed', 'brain_input']:
            for model in args.model:
                gen_dir = op.join(args.gen_res_dir, 'optuna', args.task, subject, model)
                best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']

                for itrial, trial in enumerate([0, best_trial]):
                    if input == 'reconstructed' or (input == 'brain_input' and trial != 0):
                        print(input, trial)
                        trial_dir = op.join(gen_dir, str(trial), 'eval')
                        a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + args.metric + '_' + args.clf_type + '.csv'), index_col=[0])
                        b = {}
                        b['input'] = input
                        b['accuracy'] = a[a['input'] == input]['accuracy'].mean()
                        b['subject'] = [subject]
                        b['model'] = [model]
                        b['cv'] = [0]
                        b['trial'] = 'non-optimized' if trial == 0 else 'optimized'
                        b['dir'] = gen_dir
                        b['clf'] = args.clf_type
                        results = results.append(pd.DataFrame(b))

                        # add permitations
                        # if itrial == 1: # best_trial
                        #     a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '_perm.csv'),
                        #                     index_col=[0])
                        #     for perm in range(1000):
                        #         b = {}
                        #         b['input'] = input
                        #         b['accuracy'] = a[(a['input'] == input) & (a['perm']==perm)]['accuracy'].mean()
                        #         b['subject'] = [subject]
                        #         b['model'] = [model]
                        #         b['cv'] = [0]
                        #         b['trial'] = 'permutation'
                        #         b['dir'] = gen_dir
                        #         results = results.append(pd.DataFrame(b))
    return results


def main(args):

    # plot boxplot group per subject and model
    results = load_classify_df(args)
    results.loc[results['trial']=='non-optimized', 'input'] = 'non-optimized'
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.boxplot(x='model', y='accuracy', data=results[results['trial'].isin(['optimized', 'non-optimized'])],
                hue='input', hue_order=['brain_input', 'non-optimized', 'reconstructed'],
                boxprops=dict(alpha=.7, linewidth=1), width=.9,
                whiskerprops=dict(linewidth=1),
                medianprops=dict(color='red', linewidth=1),
                ax=ax, orient='v', showfliers=False, whis=[5, 95]) # added showfliers = False, whis=[5, 95]

    # sns.boxplot(y='accuracy', data=results[results['trial']=='permutation'],
    #             boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
    #             whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
    #             medianprops=dict(linestyle=':', linewidth=.5, color='black'),
    #             capprops=dict(linestyle=':', linewidth=.5, color='black'),
    #             orient='v',
    #             ax=ax, showfliers=False, whis=[5, 95])

    sns.stripplot(x='model', y='accuracy', data=results[results['trial'].isin(['optimized', 'non-optimized'])],
                  hue='input', hue_order=['brain_input', 'non-optimized', 'reconstructed'],  size=9, orient='v',
                  jitter=.2, dodge=True, alpha=.0, linewidth=.6, linestyle='-',
                  edgecolors='black', ax=ax)

    markers = ['*', '^', 'o', 's', 'd']
    for box in range(9):
        pos = ax.get_children()[box].get_offsets().data
        for sub, mar in zip(range(pos.shape[0]), markers):
            marsize = 11 if mar == '*' else 7
            ax.plot(pos[sub][0], pos[sub][1], markersize=marsize,
                        markeredgecolor='black', color=ax.get_children()[box].get_facecolor()[0, :3],
                        marker=mar, alpha=0.5)

    #plt.ylim(.25, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if args.plot_dir != '':
        plt.savefig(op.join(args.plot_dir, 'fig5a_eval_optimized_' + args.metric + '_recon_brain_' +
                            args.clf_type +'_opt_non-opt' + '.pdf'), dpi=160,
                    transparent=True)
        plt.close()

    # reconstructed - brain input
    w = wilcoxon(results[(results['input']=='reconstructed')&(results['trial']=='optimized')]['accuracy'].values
                 - results[(results['input']=='brain_input')&(results['trial']=='optimized')]['accuracy'].values,
                 alternative='greater', zero_method='zsplit')
    print('overall effect: ', w)


    # optimized - non-optimized
    w = wilcoxon(results[(results['input']=='reconstructed')&(results['trial']=='optimized')]['accuracy'].values
                 - results[(results['input']=='non-optimized')&(results['trial']=='non-optimized')]['accuracy'].values,
                 alternative='greater', zero_method='zsplit')
    print('overall effect: ', w)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Subject to run')
    parser.add_argument('--clf_feat', '-f', type=str,  nargs="+",
                        choices=['default', 'avg_time', 'avg_chan'],
                        default=['default', 'avg_time', 'avg_chan'],
                        help='Input classifier features')
    parser.add_argument('--metric', '-x', type=str,
                        choices=['classify'],
                        default='classify',
                        help='Metric to use')
    parser.add_argument('--clf_type', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='logreg',
                        help='Type of classifier')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--gen_res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')

    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)

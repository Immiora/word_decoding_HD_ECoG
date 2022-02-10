'''
plot STOI results with permutation baseline

python figures/figure4d.py \
    --task jip_janneke \
    --type_perm shift \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --n_folds 12

'''

import sys
sys.path.insert(0, '.')

import os.path as op
import argparse
import pandas as pd
import json

import numpy as np
from scipy.stats import ttest_rel
from utils.wilcoxon import wilcoxon
from utils.private.datasets import get_subjects_by_code
from utils.general import get_stat_pval
from figures.figure4b import plot_metric_boxplot
from stats.eval_pearsonr_stoi_perm_shuffle import get_eval_input_params

def load_stoi_df(args):
    results = pd.DataFrame()
    file_tag = '_shuf_words' if args.type_perm == 'shuffle' else '_sil_only'

    for subject in args.subject:
        print(subject)
        for model in args.model:
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']

            for itrial, trial in enumerate([0, best_trial]):  # optimized and non-optimized
                trial_dir = op.join(gen_dir, str(trial), 'eval')
                input_params = get_eval_input_params(op.join(trial_dir, 'fold' + str(0)))
                words = pd.read_csv(input_params['subsets_path'])
                target_labels = words[words['subset'] == 'validation']['text'].values

                a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + args.metric + '.csv'),
                                index_col=[0]).reset_index(drop=True)
                a['subject'] = subject
                a['model'] = model
                a['trial'] = 'non-optimized' if trial == 0 else 'optimized'
                a['word'] = target_labels
                # a['successful'] = b['successful'].values if metric == 'stoi' else True # remove same or just all 1e-5?
                a['successful'] = ~(a['stoi'] == 1e-5).values
                results = results.append(a)

                # UNCOMMENT THIS WHEN PERMUTATIONS ARE DONE
                if itrial == 1:  # best_trial
                    b = pd.read_csv(
                        op.join(trial_dir,
                                'eval_n_folds' + str(args.n_folds) + '_' + args.metric + '_audio_ceil.csv'),
                        index_col=[0])
                    b['subject'] = subject
                    b['model'] = model
                    b['trial'] = 'ceiling'
                    b['successful'] = ~(b['stoi'] == 1e-5).values
                    results = results.append(pd.DataFrame(b))

                    b = pd.read_csv(
                        op.join(trial_dir,
                                'eval_n_folds' + str(args.n_folds) + '_' + args.metric + '_perm' + file_tag + '.csv'),
                        index_col=[0])
                    b['subject'] = subject
                    b['model'] = model
                    b['trial'] = 'permutation'
                    b['successful'] = ~(b['stoi'] == 1e-5).values
                    results = results.append(pd.DataFrame(b))
    return results

def main(args):
    file_tag = '_shuf_words' if args.type_perm == 'shuffle' else '_sil_only'
    results = load_stoi_df(args)

    # plot boxplots per metric
    results_all = results.copy()
    results.loc[results['successful']==False,'stoi'] = np.NaN
    plot_metric_boxplot(metric=args.metric,
                        data=results[results['trial'].isin(['optimized', 'non-optimized'])],
                        floors=results[results['trial'] == 'permutation'],
                        ceilings=results[results['trial']=='ceiling'],
                        title='fig4d_eval_' + args.metric + '_with_bounds' + file_tag + '_opt_non-opt',
                        ylim=(-.25, 1),
                        plotdir=args.plot_dir)

    # UNCOMMENT THIS WHEN PERMUTATIONS ARE DONE
    # significance
    for subject in args.subject:
        for model in args.model:
            baseline = results[(results['subject'] == subject) &
                                    (results['model'] == model) &
                                    (results['trial'] == 'permutation')][args.metric].values
            val = results[(results['trial'] == 'optimized') &
                               (results['subject'] == subject) &
                               (results['model'] == model)][args.metric].mean()
            print(subject + ' & ' + model + ' pval: ' + str(get_stat_pval(val, baseline)))


    # optimized vs non-optimized effects:
    r1 = results_all[results_all['trial'] == 'optimized'].reset_index(drop=True)
    r2 = results_all[results_all['trial'] == 'non-optimized'].reset_index(drop=True)
    ind1 = r1[r1[args.metric] == 1e-5].index
    ind2 = r2[r2[args.metric] == 1e-5].index
    r1 = r1.drop(ind1.union(ind2))
    r2 = r2.drop(ind1.union(ind2))
    w = wilcoxon(r1[args.metric] -  r2[args.metric], alternative='greater', zero_method='zsplit')
    print(args.metric, 'overall effect: ', w)
    for model in args.model:
        w = wilcoxon(r1[r1['model']==model][args.metric] - r2[r2['model']==model][args.metric],
                     alternative='greater',  zero_method='zsplit')
        print(args.metric, model, w)


    # model effects
    from scipy import stats
    import scikit_posthocs as spost
    r = results_all[results_all['trial'] == 'optimized']
    a = r[r['model'] == 'mlp'][args.metric].values
    b = r[r['model'] == 'densenet'][args.metric].values
    c = r[r['model'] == 'seq2seq'][args.metric].values
    assert len(a) == len(b) == len(c), 'unequal size arrays'
    ind1 = np.where(a == 1e-5)[0]
    ind2 = np.where(b == 1e-5)[0]
    ind3 = np.where(c == 1e-5)[0]
    ind = np.unique(np.hstack([ind1, ind2, ind3]))
    a1 = a[np.setdiff1d(range(len(a)), ind)]
    b1 = b[np.setdiff1d(range(len(a)), ind)]
    c1 = c[np.setdiff1d(range(len(a)), ind)]
    print(stats.kruskal(a1, b1, c1))
    print(spost.posthoc_dunn(np.vstack([a1, b1, c1])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--metric', '-x', type=str,
                        choices=['stoi'],
                        default='stoi',
                        help='Metric to use')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Subject to run')
    parser.add_argument('--type_perm',  type=str,
                        choices=['shift', 'shuffle'],
                        default='shift',
                        help='Model to run')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=True)

    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)

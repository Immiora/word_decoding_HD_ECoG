'''
plot VAD and STOI results with permutation baseline

python figures/figure4c.py \
    --task jip_janneke \
    --type_perm shift \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --n_folds 12

'''

import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')


import os.path as op
import argparse
import pandas as pd
import json

from os import scandir
from numpy import vstack
from scipy.stats import ttest_rel
from utils.wilcoxon import wilcoxon
from utils.private.datasets import get_subjects_by_code
from utils.general import get_stat_pval
from figures.figure4b import plot_metric_boxplot
from stats.eval_pearsonr_stoi_perm_shuffle import get_eval_input_params

def load_vad_df(args):
    file_tag = '_shuf_words' if args.type_perm == 'shuffle' else '_sil_only'

    results = pd.DataFrame()
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
                    results = results.append(pd.DataFrame(b))

                    b = pd.read_csv(
                        op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + args.metric + '_perm'+file_tag+'.csv'),
                        index_col=[0])
                    b['subject'] = subject
                    b['model'] = model
                    b['trial'] = 'permutation'
                    results = results.append(pd.DataFrame(b))
    return results

def main(args):
    if args.metric == 'vad':
        metric = 'vad_match'
    else:
        raise ValueError

    file_tag = '_shuf_words' if args.type_perm == 'shuffle' else '_sil_only'
    results = load_vad_df(args)

    # plot boxplots per metric
    plot_metric_boxplot(metric=metric,
                        data=results[results['trial'].isin(['optimized', 'non-optimized'])],
                        floors=results[results['trial'] == 'permutation'],
                        ceilings=results[results['trial']=='ceiling'],
                        title='fig4c_eval_' + metric + '_with_bounds' + file_tag + '_opt_non-opt',
                        ylim=(.2, 1.1),
                        plotdir=args.plot_dir)

    # UNCOMMENT THIS WHEN PERMUTATIONS ARE DONE
    # significance
    print(args.metric)
    for subject in args.subject:
        for model in args.model:
            baseline = results[(results['subject'] == subject) &
                                    (results['model'] == model) &
                                    (results['trial'] == 'permutation')][metric].values
            val = results[(results['trial'] == 'optimized') &
                               (results['subject'] == subject) &
                               (results['model'] == model)][metric].mean()
            print(subject + ' & ' + model + ' pval: ' + str(get_stat_pval(val, baseline)))


    # optimized vs non-optimized effects:
    w = wilcoxon(results[results['trial'] == 'optimized'][metric] -
             results[results['trial'] == 'non-optimized'][metric], alternative='greater',
             zero_method='zsplit')
    print(metric, 'overall effect: ', w)

    for model in args.model:
        w = wilcoxon(results[(results['trial'] == 'optimized') & (results['model']==model)][metric] -
                 results[(results['trial'] == 'non-optimized') & (results['model']==model)][metric], alternative='greater',
                 zero_method='zsplit')
        print(metric, model, w)

    # model effects
    from scipy import stats
    import scikit_posthocs as spost
    r = results[results['trial'] == 'optimized']
    a = r[r['model'] == 'mlp'][metric]
    b = r[r['model'] == 'densenet'][metric]
    c = r[r['model'] == 'seq2seq'][metric]
    print(stats.kruskal(a, b, c))
    print(spost.posthoc_dunn(vstack([a, b, c])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--metric', '-x', type=str,
                        choices=['vad'],
                        default='vad',
                        help='Metric to use')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Subject to run')
    parser.add_argument('--type_perm',  type=str,
                        choices=['shift'],
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

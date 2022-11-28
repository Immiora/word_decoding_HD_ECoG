'''
Plot correlations


python figures/figure4b.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --type_perm shift \
    --n_folds 12
'''

import sys
sys.path.insert(0, '.')

import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
import json
import seaborn as sns

from scipy.stats import ttest_rel
from utils.wilcoxon import wilcoxon
from utils.general import get_stat_pval
from utils.private.datasets import get_subjects_by_code
from matplotlib import pyplot as plt
from evaluate_decoders.eval_pearsonr import calculate_pearsonr_flattened
from stats.eval_pearsonr_stoi_perm_shuffle import load_t_moments, normalize, get_eval_input_params


def plot_metric_boxplot(metric, data, floors=None, ceilings=None, title='', ylim = (0, 1), plotdir=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4)) # changed from 5,3 to 6,3

    if floors is not None:
        sns.boxplot(y=metric, data=floors,
                    orient='v',
                    boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    showfliers=False, whis=[5, 95], ax=ax)

    if ceilings is not None:
        sns.boxplot(y=metric, data=ceilings,
                    orient='v',
                    boxprops=dict(alpha=.3, linewidth=.5, facecolor='salmon'),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    ax=ax, color='gray', showfliers=False, whis=[5, 95])

    sns.stripplot(x='model', y=metric, data=data,
                  hue='trial',
                  size=9, orient='v',
                  jitter=.2, dodge=True, alpha=.0, linewidth=.6,
                  edgecolors='black', ax=ax)

    sns.boxplot(x='model', y=metric, data=data, hue='trial',
                boxprops=dict(alpha=.7), width=.7,
                medianprops=dict(color='red'),
                ax=ax, orient='v', showfliers=False, whis=[5, 95])

    markers = ['*', '^', 'o', 's', 'd']
    colors = sns.color_palette()[:3]
    n = len(data['subject'].unique())
    box = 0
    for model in range(3):
        for trial in range(2):
            print(box)
            for sub, mar in zip(range(n), markers):
                pos = ax.get_children()[box].get_offsets().data
                print(len(pos))
                marsize = 11 if mar == '*' else 7  # was 11 and 15: too big
                ax.plot(np.nanmean(pos[:, 0])+0.08*(sub-2),
                        np.nanmedian(pos[sub*12:sub*12+12, 1]), markersize=marsize+4,
                        markeredgecolor='black', color=colors[model], marker=mar)
                for i in range(12): # replace with unique folds?
                    ax.plot(np.mean(pos[:, 0])+0.08*(sub-2), pos[sub*12+i][1], markersize=marsize,
                            markeredgecolor='black', color=colors[model], marker=mar, alpha=0.1)
            box = box + 1

    plt.ylim(ylim)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if plotdir is not None:
        plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
        plt.close()

def load_pearson_df(args):
    res_pearsonr, qs = pd.DataFrame(), pd.DataFrame()
    file_tag = '_shuf_words' if args.type_perm == 'shuffle' else '_sil_only'
    for isubj, subject in enumerate(args.subject):
        print(subject)
        for imod, model in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']

            for itrial, trial in enumerate([0, best_trial]): # optimized and non-optimized
                trial_dir = op.join(gen_dir, str(trial), 'eval')

                for fold in range(12):
                    fold_dir = op.join(trial_dir, 'fold' + str(fold))
                    assert op.exists(fold_dir), 'No fold dir at ' + fold_dir

                    val_predictions = np.load(os.path.join(fold_dir, 'val_predictions.npy'))
                    val_targets = np.load(os.path.join(fold_dir, 'val_targets.npy'))
                    input_params = get_eval_input_params(fold_dir)
                    t_mean, t_std = load_t_moments(input_params)
                    val_predictions_z = normalize(val_predictions, t_mean, t_std)
                    val_targets_z = normalize(val_targets, t_mean, t_std)
                    words = pd.read_csv(op.join(fold_dir, op.basename(input_params['subsets_path_fold'])), index_col=[0])
                    target_label = words[words['subset']=='validation']['text'].values[0]

                    a = pd.DataFrame({'subject':[], 'model':[], 'trial':[], 'fold':[], 'pearsonr':[]})
                    a.loc[0, 'subject'] = subject
                    a['model'] = model
                    a['trial'] = 'non-optimized' if trial == 0 else 'optimized'
                    a['fold'] = fold
                    a['pearsonr'] = calculate_pearsonr_flattened(val_predictions_z, val_targets_z)['r'].values[0]
                    a['word'] = target_label
                    res_pearsonr = res_pearsonr.append(a)

                # UNCOMMENT THIS WHEN PERMUTATIONS ARE DONE
                if itrial == 1:  # best_trial
                    b = pd.read_csv(
                        op.join(trial_dir,
                                'eval_n_folds' + str(args.n_folds) + '_pearsonr'+ '_audio_ceil.csv'),
                        index_col=[0])
                    b['subject'] = subject
                    b['model'] = model
                    b['trial'] = 'ceiling'
                    b['pearsonr'] = b['r']
                    res_pearsonr = res_pearsonr.append(pd.DataFrame(b))

                    b = pd.read_csv(
                        op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_pearsonr'+'_perm'+file_tag+'.csv'),
                        index_col=[0])
                    b['subject'] = subject
                    b['model'] = model
                    b['trial'] = 'permutation'
                    res_pearsonr = res_pearsonr.append(b)
                    x = {'subject': subject, 'model':model, 'q':b['pearsonr'].quantile(0.95)}
                    qs = qs.append(x, ignore_index=True)

    return res_pearsonr

def main(args):
    res_pearsonr = load_pearson_df(args)
    file_tag = '_shuf_words' if args.type_perm == 'shuffle' else '_sil_only'

    # plot
    plot_metric_boxplot(metric='pearsonr',
                        data=res_pearsonr[res_pearsonr['trial'].isin(['optimized', 'non-optimized'])],
                        ceilings=res_pearsonr[res_pearsonr['trial']=='ceiling'],
                        floors=res_pearsonr[res_pearsonr['trial'] == 'permutation'],
                        title='fig4b_eval_pearsonr' + '_with_bounds' + file_tag + '_opt_non-opt_ceil',
                        ylim=(-.25, 1),
                        plotdir=args.plot_dir)


    # significance based on permutation baseline
    for subject in args.subject:
        for model in args.model:
            baseline = res_pearsonr[(res_pearsonr['subject']==subject) &
                                    (res_pearsonr['model']==model) &
                                    (res_pearsonr['trial']=='permutation')]['pearsonr'].values
            val = res_pearsonr[(res_pearsonr['trial'] == 'optimized') &
                               (res_pearsonr['subject'] == subject) &
                               (res_pearsonr['model'] == model)]['pearsonr'].mean()
            print(subject + ' & ' + model + ' pval: ' + str(get_stat_pval(val, baseline)))


    # optimized vs non-optimized effects:
    w = wilcoxon(res_pearsonr[res_pearsonr['trial'] == 'optimized']['pearsonr'] -
             res_pearsonr[res_pearsonr['trial'] == 'non-optimized']['pearsonr'], alternative='greater',
             zero_method='zsplit')
    print('overall effect: ', w)

    for model in args.model:
        w = wilcoxon(res_pearsonr[(res_pearsonr['trial'] == 'optimized') & (res_pearsonr['model']==model)]['pearsonr'] -
                 res_pearsonr[(res_pearsonr['trial'] == 'non-optimized') & (res_pearsonr['model']==model)]['pearsonr'], alternative='greater',
                 zero_method='zsplit')
        print(model, w)


    # model effects
    from scipy import stats
    import scikit_posthocs as spost
    r = res_pearsonr[res_pearsonr['trial'] == 'optimized']
    a = r[r['model'] == 'mlp']['pearsonr']
    b = r[r['model'] == 'densenet']['pearsonr']
    c = r[r['model'] == 'seq2seq']['pearsonr']
    print(stats.kruskal(a, b, c))
    print(spost.posthoc_dunn(np.vstack([a, b, c])))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop', 'fvxs'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Model to run')
    parser.add_argument('--type_perm',  type=str,
                        choices=['shift', 'shuffle'],
                        default='shift',
                        help='Model to run')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=True)
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)

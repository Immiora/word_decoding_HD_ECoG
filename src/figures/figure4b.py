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
from stats.eval_pearsonr_stoi_perm_shuffle import load_t_moments, normalize

def plot_metric_boxplot(metric, data, floors=None, ceilings=None, title='', ylim = (0, 1), plot_dots=False, plotdir=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3)) # changed from 5,3 to 6,3

    if floors is not None:
        sns.boxplot(x='subject', y=metric, data=floors,
                    hue='model',
                    orient='v',
                    boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    showfliers=False, whis=[5, 95], ax=ax)

    if ceilings is not None:
        sns.boxplot(x='subject', y=metric, data=ceilings,
                    hue='model',
                    orient='v',
                    boxprops=dict(alpha=.3, linewidth=.5, facecolor='salmon'),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    ax=ax, color='gray', showfliers=False, whis=[5, 95])

    if plot_dots:
        sns.stripplot(x='subject', y=metric, data=data,
                      hue='model', size=6, orient='v', # changed size from 4 to 6
                      jitter=.2, dodge=True, alpha=.4, linewidth=.6,
                      edgecolors='black', ax=ax)

    sns.boxplot(x='subject', y=metric, data=data,
                boxprops=dict(alpha=.7, linewidth=.5),
                whiskerprops=dict(linewidth=1),
                medianprops=dict(color='red', linewidth=.5),
                capprops=dict(linewidth=.5),
                showfliers=False, whis=[5, 95],  # added showfliers = False, whis=[5, 95]
                hue='model',
                orient='v')

    sns.boxplot(x='model', y=metric, data=data,
                boxprops=dict(alpha=.7, linewidth=.5),
                whiskerprops=dict(linewidth=1),
                medianprops=dict(color='red', linewidth=.5),
                capprops=dict(linewidth=.5),
                showfliers=False, whis=[5, 95],  # added showfliers = False, whis=[5, 95]
                hue='subject',
                orient='v')

    fig, ax = plt.subplots(1, 1, figsize=(6, 3)) # changed from 5,3 to 6,3
    sns.stripplot(x='model', y=metric, data=data,
                  hue='subject',
                  size=9, orient='v',
                  jitter=0, dodge=True, alpha=.0, linewidth=.6,
                  edgecolors='black', ax=ax)

    sns.boxplot(x='model', y=metric, data=data,
                boxprops=dict(alpha=.7, facecolor='none'),
                medianprops=dict(color='red'),
                ax=ax, orient='v', showfliers=False, whis=[5, 95])  # added showfliers = False, whis=[5, 95]

    markers = ['*', '^', 'o', 's', 'd']
    colors = sns.color_palette()[:3]
    n = len(args.subject)
    box = 0
    for model in range(3):
        print(box)
        for sub, mar in zip(range(n), markers):
            pos = ax.get_children()[box].get_offsets().data
            marsize = 11 if mar == '*' else 7  # was 11 and 15: too big
            ax.plot(np.mean(pos[:, 0]),
                    np.median(pos[:, 1]), markersize=marsize+5,
                    markeredgecolor='black', color=colors[model], marker=mar)
            for i in range(12):
                ax.plot(pos[i][0], pos[i][1], markersize=marsize,
                        markeredgecolor='black', color=colors[model], marker=mar, alpha=0.1)
            box = box + 1

    plt.ylim(ylim)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if plotdir is not None:
        plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
        plt.close()

def main(args):
    for type_perm in args.type_perm:
        res_pearsonr, qs = pd.DataFrame(), pd.DataFrame()
        file_tag = '_shuf_words' if type_perm == 'shuffle' else '_sil_only'

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
                        t_mean, t_std = load_t_moments(fold_dir)
                        val_predictions_z = normalize(val_predictions, t_mean, t_std)
                        val_targets_z = normalize(val_targets, t_mean, t_std)

                        a = pd.DataFrame({'subject':[], 'model':[], 'trial':[], 'fold':[], 'pearsonr':[]})
                        a.loc[0, 'subject'] = subject
                        a['model'] = model
                        a['trial'] = 'non-optimized' if trial == 0 else 'optimized'
                        a['fold'] = fold
                        a['pearsonr'] = calculate_pearsonr_flattened(val_predictions_z, val_targets_z)['r'].values[0]
                        res_pearsonr = res_pearsonr.append(a)

                    # UNCOMMENT THIS WHEN PERMUTATIONS ARE DONE
                    if itrial == 1:  # best_trial
                        b = pd.read_csv(
                            op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_pearsonr'+'_perm'+file_tag+'.csv'),
                            index_col=[0])
                        b['subject'] = subject
                        b['model'] = model
                        b['trial'] = 'permutation'
                        res_pearsonr = res_pearsonr.append(b)
                        x = {'subject': subject, 'model':model, 'q':b['pearsonr'].quantile(0.95)}
                        qs = qs.append(x, ignore_index=True)

        # UNCOMMENT THIS WHEN PERMUTATIONS ARE DONE

        plot_metric_boxplot(metric='pearsonr',
                            data=res_pearsonr[res_pearsonr['trial'] == 'optimized'],
                            ceilings=None,
                            floors=res_pearsonr[res_pearsonr['trial'] == 'permutation'],
                            title='fig4b_eval_pearsonr_with' + '_perm' + file_tag,
                            ylim=(-.25, 1),
                            plot_dots=args.plot_dots,
                            plotdir=args.plot_dir)


        # significance based on permutation baseline
        print(type_perm)
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
        # overall effect: WilcoxonResult(w_statistic=12690.0, z_statistic=6.492476217927286, pvalue=4.221842572411459e-11)


        for model in args.model:
            w = wilcoxon(res_pearsonr[(res_pearsonr['trial'] == 'optimized') & (res_pearsonr['model']==model)]['pearsonr'] -
                     res_pearsonr[(res_pearsonr['trial'] == 'non-optimized') & (res_pearsonr['model']==model)]['pearsonr'], alternative='greater',
                     zero_method='zsplit')
            print(model, w)

            # mlp
            # WilcoxonResult(w_statistic=1277.0, z_statistic=2.6649002865887965, pvalue=0.0038505576830898413)
            # densenet
            # WilcoxonResult(w_statistic=1710.0, z_statistic=5.852474386293075, pvalue=2.4215642596013072e-09)
            # seq2seq
            # WilcoxonResult(w_statistic=1141.0, z_statistic=1.6637222783675911, pvalue=0.048083971023011554)

        # model effects
        for model in args.model:
            # w = wilcoxon(results_all[(results_all['trial'] == 'optimized') & (results_all['model']==model)][col] -
            #          results_all[(results_all['trial'] == 'non-optimized') & (results_all['model']==model)][col], alternative='greater',
            #          zero_method='zsplit')
            w = wilcoxon(data[data['model'] == model]['pearsonr'] - data[data['model'] == model]['pearsonr'],
                         alternative='greater', zero_method='zsplit')
            # w = wilcoxon(results[(results['trial'] == 'optimized') & (results['model']==model)][col] -
            #          results[(results['trial'] == 'non-optimized') & (results['model']==model)][col], alternative='greater',
            #          zero_method='zsplit')
            print(model, w)


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
                        help='Model t34o run')
    parser.add_argument('--type_perm',  type=str,  nargs="+",
                        choices=['shift', 'shuffle'],
                        default=['shift', 'shuffle'],
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

# sns.barplot(x='q', y='model', data=qs,
#             hue='subject',
#             orient='h',
#             color='gray',
#             linewidth=1,
#             ax=ax)